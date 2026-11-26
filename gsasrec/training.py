import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


def make_batch(histories, max_length):
    """Build right-padded inputs and next-item targets."""
    sequences = [
        torch.tensor(history[-(max_length + 1):], dtype=torch.long)
        for history in histories
        if len(history) >= 2
    ]
    if not sequences:
        raise ValueError("A batch needs at least one history with two items.")

    inputs = pad_sequence([seq[:-1] for seq in sequences], batch_first=True)
    targets = pad_sequence([seq[1:] for seq in sequences], batch_first=True)
    return inputs, targets


def sample_uniform_negatives(positives, num_items, num_negatives):
    """Draw uniformly with replacement, skipping each current positive ID."""
    draws = torch.randint(
        1,
        num_items,
        (*positives.shape, num_negatives),
        device=positives.device,
    )
    return draws + (draws >= positives.unsqueeze(-1))


def gsasrec_loss(model, inputs, targets, num_items, num_negatives=32, t=0.5):
    if not 1 <= num_negatives <= num_items - 1:
        raise ValueError("Expected between 1 and num_items - 1 negatives.")
    if not 0.0 <= t <= 1.0:
        raise ValueError("Expected t between 0 and 1.")

    valid = targets.ne(0)
    states = model(inputs)[valid].unsqueeze(1)
    positives = targets[valid]
    negatives = sample_uniform_negatives(positives, num_items, num_negatives)
    candidate_ids = torch.cat((positives.unsqueeze(-1), negatives), dim=-1)
    logits = model.score_candidates(states, candidate_ids.unsqueeze(1)).squeeze(1)

    alpha = num_negatives / (num_items - 1)
    beta = 1.0 - t * (1.0 - alpha)
    positive_loss = beta * F.softplus(-logits[:, 0])
    negative_loss = F.softplus(logits[:, 1:]).sum(dim=1)
    return ((positive_loss + negative_loss) / (num_negatives + 1)).mean()