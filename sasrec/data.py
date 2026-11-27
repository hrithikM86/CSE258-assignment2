"""Original SASRec sequence alignment and training-only negative sampling."""
import numpy as np
import torch


def left_pad(histories, max_length):
    if max_length < 1 or not histories or any(not h for h in histories):
        raise ValueError("Expected nonempty histories and positive max_length.")
    batch = np.zeros((len(histories), max_length), dtype=np.int64)
    for row, history in enumerate(histories):
        tail = history[-max_length:]
        batch[row, -len(tail):] = tail
    return torch.from_numpy(batch)


class TrainingSampler:
    """Sample users with replacement, then one unseen negative per target.

    Only training histories are supplied. Held-out validation/test identities
    are deliberately unavailable to the sampler.
    """
    def __init__(self, histories, num_items, max_length=50, seed=42):
        if num_items < 2 or max_length < 1:
            raise ValueError("Expected at least two items and positive max_length.")
        self.histories = histories
        self.users = np.array([u for u, h in histories.items() if len(h) >= 2])
        if not len(self.users):
            raise ValueError("Training requires a user with at least two items.")
        self.seen = {u: set(histories[u]) for u in self.users}
        for user, seen in self.seen.items():
            if not seen or min(seen) < 1 or max(seen) > num_items:
                raise ValueError("Training item IDs are outside the catalog.")
            if len(seen) == num_items:
                raise ValueError(f"No unseen training negatives for user {user}.")
        self.num_items = num_items
        self.max_length = max_length
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size):
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        users = self.rng.choice(self.users, size=batch_size, replace=True)
        shape = (batch_size, self.max_length)
        inputs = np.zeros(shape, dtype=np.int64)
        positives = np.zeros(shape, dtype=np.int64)
        negatives = np.zeros(shape, dtype=np.int64)
        for row, user in enumerate(users):
            history = self.histories[user]
            x = history[:-1][-self.max_length:]
            y = history[1:][-self.max_length:]
            inputs[row, -len(x):] = x
            positives[row, -len(y):] = y
            for col in range(self.max_length-len(y), self.max_length):
                draw = int(self.rng.integers(1, self.num_items+1))
                while draw in self.seen[user]:
                    draw = int(self.rng.integers(1, self.num_items+1))
                negatives[row, col] = draw
        return tuple(torch.from_numpy(a) for a in (inputs, positives, negatives))
