import torch
from torch import nn


class GSASRec(nn.Module):
    def __init__(
        self,
        num_items,
        max_length=200,
        hidden_size=128,
        num_heads=1,
        num_layers=2,
        dropout=0.5,
    ):
        super().__init__()
        self.max_length = max_length
        self.item_embeddings = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        self.output_embeddings = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size),
            enable_nested_tensor=False,
        )

    def forward(self, item_ids):
        """Encode sequences with padding zeros on the right."""
        length = item_ids.size(1)
        if not 1 <= length <= self.max_length:
            raise ValueError("Sequence length is outside the supported range.")

        positions = torch.arange(length, device=item_ids.device)
        padded = item_ids.eq(0).unsqueeze(-1)
        inputs = self.item_embeddings(item_ids) + self.position_embeddings(positions)
        inputs = self.dropout(inputs).masked_fill(padded, 0)

        # Block attention to future items.
        causal_mask = torch.ones(
            length, length, dtype=torch.bool, device=item_ids.device
        ).triu(1)
        states = self.encoder(inputs, mask=causal_mask)
        return states.masked_fill(padded, 0)

    def score_candidates(self, states, candidate_ids):
        """Score candidate IDs at each sequence position."""
        candidates = self.output_embeddings(candidate_ids)
        return torch.einsum("bth,btkh->btk", states, candidates)