"""PyTorch port of the original SASRec model and attention operations.

Reference: https://github.com/kang205/SASRec (model.py and modules.py).
Modified implementation: TensorFlow operations replaced by explicit PyTorch
layers, with the original normalization/residual order and shared item table.
The upstream attention module credits Kyubyong Park (June 2017),
https://github.com/Kyubyong/transformer. See LICENSE-SASRec.txt.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class SASRecBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.feedforward_in = nn.Linear(hidden_size, hidden_size)
        self.feedforward_out = nn.Linear(hidden_size, hidden_size)
        self.feedforward_dropout = nn.Dropout(dropout)

    def forward(self, sequence):
        batch, length, hidden = sequence.shape
        # The authors normalize queries only; keys and values use raw states.
        queries = self.attention_norm(sequence)
        def split_heads(tensor):
            return tensor.reshape(batch, length, self.num_heads, self.head_size).transpose(1, 2)
        q = split_heads(self.query(queries))
        k = split_heads(self.key(sequence))
        v = split_heads(self.value(sequence))
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        causal = torch.ones(length, length, device=sequence.device, dtype=torch.bool).tril()
        # Match the reference's state-derived key and query masks. A finite
        # sentinel also avoids NaNs at the all-masked left-padding positions.
        key_mask = sequence.abs().sum(-1).ne(0)
        allowed = causal[None, None] & key_mask[:, None, None, :]
        weights = torch.softmax(logits.masked_fill(~allowed, -(2**32) + 1), dim=-1)
        query_mask = queries.abs().sum(-1).ne(0)
        weights = weights * query_mask[:, None, :, None]
        attended = self.attention_dropout(weights) @ v
        attended = attended.transpose(1, 2).reshape(batch, length, hidden)
        # No attention output projection in the original implementation.
        sequence = queries + attended
        normalized = self.feedforward_norm(sequence)
        feedforward = self.feedforward_dropout(F.relu(self.feedforward_in(normalized)))
        feedforward = self.feedforward_dropout(self.feedforward_out(feedforward))
        return normalized + feedforward


class SASRec(nn.Module):
    def __init__(self, num_items, max_length=50, hidden_size=50,
                 num_blocks=2, num_heads=1, dropout=0.5):
        super().__init__()
        if min(num_items, max_length, hidden_size, num_blocks, num_heads) < 1:
            raise ValueError("Model sizes must be positive.")
        if hidden_size % num_heads or not 0 <= dropout < 1:
            raise ValueError("Invalid head count or dropout.")
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.item_embeddings = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_size, num_heads, dropout) for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        # tf.get_variable and tf.layers dense/conv1d use Glorot uniform;
        # torch's default embedding/linear initializers would differ.
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.item_embeddings.weight[0].zero_()

    def forward(self, item_ids):
        """Encode fixed-length, LEFT-padded histories; last column is newest."""
        if item_ids.ndim != 2 or item_ids.size(1) != self.max_length:
            raise ValueError(f"Expected [batch, {self.max_length}] left-padded input.")
        mask = item_ids.ne(0).unsqueeze(-1)
        positions = torch.arange(self.max_length, device=item_ids.device)
        sequence = self.item_embeddings(item_ids) * math.sqrt(self.hidden_size)
        sequence = self.embedding_dropout(sequence + self.position_embeddings(positions))
        sequence = sequence * mask
        for block in self.blocks:
            sequence = block(sequence) * mask
        return self.final_norm(sequence)

    def score_candidates(self, states, candidate_ids):
        """Dot products with the SAME item table used for sequence inputs."""
        return (states.unsqueeze(-2) * self.item_embeddings(candidate_ids)).sum(-1)
