import torch
from torch import nn

from gsasrec_text.model import TextGSASRec


class TextRatingGSASRec(TextGSASRec):
    def __init__(self, num_items, embeddings_path, **model_options):
        super().__init__(
            num_items=num_items,
            embeddings_path=embeddings_path,
            **model_options,
        )
        self.rating_embeddings = nn.Embedding(
            num_embeddings=6,
            embedding_dim=self.item_embeddings.embedding_dim,
            padding_idx=0,
        )

    def forward(self, item_ids, rating_ids):
        """Encode right-padded histories with aligned ratings."""
        if item_ids.shape != rating_ids.shape:
            raise ValueError("Item IDs and rating IDs must have the same shape.")

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

        ratings = self.rating_embeddings(rating_ids)
        inputs = (
            learned_items
            + projected_text
            + ratings
            + self.position_embeddings(positions)
        )
        inputs = self.dropout(inputs).masked_fill(padded, 0)

        causal_mask = torch.ones(
            length,
            length,
            dtype=torch.bool,
            device=item_ids.device,
        ).triu(1)

        states = self.encoder(inputs, mask=causal_mask)
        return states.masked_fill(padded, 0)