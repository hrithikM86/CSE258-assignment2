"""One-positive/one-negative binary cross-entropy from SASRec Eq. (6)."""
import torch
from torch.nn import functional as F


def sasrec_loss(model, inputs, positives, negatives, l2_emb=0.0):
    if inputs.shape != positives.shape or inputs.shape != negatives.shape:
        raise ValueError("Inputs, positives, and negatives must be aligned.")
    if l2_emb < 0:
        raise ValueError("l2_emb must be nonnegative.")
    valid = positives.ne(0)
    if not valid.any():
        raise ValueError("The batch has no next-item targets.")
    states = model(inputs)[valid]
    pos_logits = (states * model.item_embeddings(positives[valid])).sum(-1)
    neg_logits = (states * model.item_embeddings(negatives[valid])).sum(-1)
    loss = (F.softplus(-pos_logits) + F.softplus(neg_logits)).mean()
    if l2_emb:
        loss = loss + 0.5 * l2_emb * (
            model.item_embeddings.weight[1:].square().sum()
            + model.position_embeddings.weight.square().sum()
        )
    return loss
