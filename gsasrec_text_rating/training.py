import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from gsasrec.training import sample_uniform_negatives


def make_batch(item_histories, rating_histories, max_length):
    """Build aligned inputs and next-item targets with right padding."""
    input_items = []
    input_ratings = []
    target_items = []

    for items, ratings in zip(item_histories, rating_histories, strict=True):
        if len(items) != len(ratings):
            raise ValueError("Each item history needs an equally long rating history.")
        if len(items) < 2:
            continue

        items = torch.tensor(items[-(max_length + 1):], dtype=torch.long)
        ratings = torch.tensor(ratings[-(max_length + 1):], dtype=torch.long)

        input_items.append(items[:-1])
        input_ratings.append(ratings[:-1])
        target_items.append(items[1:])

    if not input_items:
        raise ValueError("A batch needs at least one history with two items.")

    return (
        pad_sequence(input_items, batch_first=True),
        pad_sequence(input_ratings, batch_first=True),
        pad_sequence(target_items, batch_first=True),
    )


def gsasrec_loss(
    model,
    inputs,
    ratings,
    targets,
    num_items,
    num_negatives=32,
    t=0.5,
):
    if not 1 <= num_negatives <= num_items - 1:
        raise ValueError("Expected between 1 and num_items - 1 negatives.")
    if not 0.0 <= t <= 1.0:
        raise ValueError("Expected t between 0 and 1.")

    valid = targets.ne(0)
    states = model(inputs, ratings)[valid].unsqueeze(1)
    positives = targets[valid]

    negatives = sample_uniform_negatives(
        positives, num_items, num_negatives
    )
    candidate_ids = torch.cat(
        (positives.unsqueeze(-1), negatives), dim=-1
    )
    logits = model.score_candidates(
        states, candidate_ids.unsqueeze(1)
    ).squeeze(1)

    alpha = num_negatives / (num_items - 1)
    beta = 1.0 - t * (1.0 - alpha)

    positive_loss = beta * F.softplus(-logits[:, 0])
    negative_loss = F.softplus(logits[:, 1:]).sum(dim=1)
    return (
        (positive_loss + negative_loss) / (num_negatives + 1)
    ).mean()