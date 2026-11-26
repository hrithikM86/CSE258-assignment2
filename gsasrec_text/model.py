import numpy as np
import torch
from torch import nn

from gsasrec.model import GSASRec


class TextGSASRec(GSASRec):
    def __init__(self, num_items, embeddings_path, **model_options):
        super().__init__(num_items=num_items, **model_options)

        with np.load(embeddings_path, allow_pickle=False) as data:
            text_vectors = torch.from_numpy(data["embeddings"]).float()
            has_text = torch.from_numpy(data["has_text"]).bool()

        expected_shape = (num_items + 1, 384)
        if text_vectors.shape != expected_shape:
            raise ValueError(
                f"Expected text embeddings of shape {expected_shape}, "
                f"got {tuple(text_vectors.shape)}."
            )
        if has_text.shape != (num_items + 1,) or has_text[0].item():
            raise ValueError("Expected one text flag per item and False for padding.")

        # Frozen inputs move with the model and are loaded from the shared file.
        self.register_buffer("text_vectors", text_vectors, persistent=False)
        self.register_buffer("has_text", has_text, persistent=False)
        self.text_projection = nn.Linear(
            text_vectors.shape[1], self.item_embeddings.embedding_dim, bias=False
        )

    def forward(self, item_ids):
        """Encode sequences with padding zeros on the right."""
        length = item_ids.size(1)
        if not 1 <= length <= self.max_length:
            raise ValueError("Sequence length is outside the supported range.")

        positions = torch.arange(length, device=item_ids.device)
        padded = item_ids.eq(0).unsqueeze(-1)

        learned_items = self.item_embeddings(item_ids)
        projected_text = self.text_projection(self.text_vectors[item_ids])
        projected_text = projected_text.masked_fill(
            ~self.has_text[item_ids].unsqueeze(-1), 0
        )

        inputs = learned_items + projected_text + self.position_embeddings(positions)
        inputs = self.dropout(inputs).masked_fill(padded, 0)

        causal_mask = torch.ones(
            length, length, dtype=torch.bool, device=item_ids.device
        ).triu(1)
        states = self.encoder(inputs, mask=causal_mask)
        return states.masked_fill(padded, 0)