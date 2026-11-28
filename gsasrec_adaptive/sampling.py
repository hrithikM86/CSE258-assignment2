"""Fixed-budget negative mixtures; eligibility matches the copied uniform control.

Only padding and the current positive are excluded, as in Hrithik's baseline.
Other user positives can therefore appear as negatives in ALL compared modes.
"""
import torch
from gsasrec_adaptive.training import sample_uniform_negatives


class SemanticSampler:
    def __init__(self, num_items, neighbors, has_text, fraction=0.5, seed=42):
        if not 0 <= fraction < 1:
            raise ValueError("Keep a nonzero uniform component: 0 <= fraction < 1.")
        self.num_items = num_items
        self.neighbors = torch.as_tensor(neighbors, dtype=torch.long)
        self.has_text = torch.as_tensor(has_text, dtype=torch.bool)
        if self.neighbors.ndim != 2 or self.neighbors.shape[0] != num_items+1:
            raise ValueError("Neighbor table does not match the catalog.")
        if self.has_text.shape != (num_items+1,) or self.has_text[0]:
            raise ValueError("Invalid text flags.")
        if self.neighbors.min() < 0 or self.neighbors.max() > num_items:
            raise ValueError("Invalid neighbor IDs.")
        usable = self.neighbors[self.has_text]
        anchors = torch.arange(num_items+1)[self.has_text,None]
        if usable.numel() and (usable.eq(0).any() or usable.eq(anchors).any()):
            raise ValueError("Semantic neighbors include padding or their anchor.")
        self.fraction = fraction
        self.generator = torch.Generator(device="cpu").manual_seed(seed+104729)
        self.reset_stats()

    def reset_stats(self):
        self.stats = dict(total_negatives=0, requested_semantic=0,
                          actual_semantic=0, fallback_negatives=0)

    def sample(self, positives, num_negatives):
        # Always draw the same full uniform budget first. Extra semantic RNG
        # is isolated, so it does not consume the model/dropout RNG stream.
        draws = sample_uniform_negatives(positives,self.num_items,num_negatives)
        count = min(num_negatives-1, int(num_negatives*self.fraction+0.5))
        self.stats["total_negatives"] += draws.numel()
        self.stats["requested_semantic"] += positives.numel()*count
        if count:
            anchors = positives.detach().cpu()
            present = self.has_text[anchors]
            rows = present.nonzero(as_tuple=True)[0]
            offsets = torch.randint(self.neighbors.shape[1], (len(rows),count),generator=self.generator)
            semantic = self.neighbors[anchors[rows,None],offsets].to(draws.device)
            device_rows = rows.to(draws.device)
            draws[device_rows,-count:] = semantic
            actual = len(rows)*count
            self.stats["actual_semantic"] += actual
            self.stats["fallback_negatives"] += (positives.numel()-len(rows))*count
        return draws
