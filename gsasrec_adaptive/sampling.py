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

    def sample(self, positives, num_negatives, model=None, states=None):
        draws = sample_uniform_negatives(positives,self.num_items,num_negatives)
        counts = self.semantic_counts(positives,draws,model,states)
        counts = counts.clamp(min=0,max=num_negatives-1).cpu()
        self.stats["total_negatives"] += draws.numel()
        self.stats["requested_semantic"] += counts.sum().item()
        maximum = int(counts.max().item()) if counts.numel() else 0
        if maximum:
            anchors = positives.detach().cpu()
            present = self.has_text[anchors]
            rows = (present & counts.gt(0)).nonzero(as_tuple=True)[0]
            offsets = torch.randint(self.neighbors.shape[1], (len(rows),maximum),generator=self.generator)
            semantic = self.neighbors[anchors[rows,None],offsets].to(draws.device)
            device_rows = rows.to(draws.device)
            original = draws[device_rows,-maximum:]
            mask = torch.arange(maximum)[None,:] >= maximum-counts[rows,None]
            draws[device_rows,-maximum:] = torch.where(mask.to(draws.device),semantic,original)
            actual = counts[rows].sum().item()
            self.stats["actual_semantic"] += actual
            self.stats["fallback_negatives"] += counts.sum().item()-actual
        return draws

    def semantic_counts(self,positives,uniform_draws,model=None,states=None):
        count = int(uniform_draws.size(1)*self.fraction+0.5)
        return torch.full(positives.shape,count,dtype=torch.long)


class AdaptiveSemanticSampler(SemanticSampler):
    """Epoch curriculum or per-target score-margin adaptation, no extra forward."""
    def __init__(self,*args,mode="curriculum",warmup_epochs=3,ramp_epochs=10,**kwargs):
        super().__init__(*args,**kwargs)
        if mode not in ("curriculum","confidence") or warmup_epochs<0 or ramp_epochs<1:
            raise ValueError("Invalid adaptive schedule.")
        self.mode=mode
        self.warmup_epochs=warmup_epochs
        self.ramp_epochs=ramp_epochs
        self.epoch=1

    def begin_epoch(self,epoch):
        if epoch<1:
            raise ValueError("Epoch must be positive.")
        self.epoch=epoch
        self.reset_stats()
        self.stats.update(confidence_sum=0.0,confidence_targets=0)

    def semantic_counts(self,positives,uniform_draws,model=None,states=None):
        if self.epoch<=self.warmup_epochs:
            return torch.zeros(positives.shape,dtype=torch.long)
        if self.mode=="curriculum":
            progress=min(1.,(self.epoch-self.warmup_epochs)/self.ramp_epochs)
            count=int(uniform_draws.size(1)*self.fraction*progress+0.5)
            return torch.full(positives.shape,count,dtype=torch.long)
        if model is None or states is None:
            raise ValueError("Confidence sampling needs current model states.")
        with torch.no_grad():
            positive_logits=(states.detach()*model.output_embeddings(positives)).sum(-1)
            negative_logits=(states.detach()[:,None,:]*model.output_embeddings(uniform_draws)).sum(-1)
            confidence=torch.sigmoid(positive_logits-negative_logits.max(-1).values)
            # Below the strongest uniform negative => no semantic allocation.
            fraction=self.fraction*(2*confidence-1).clamp(0,1)
            counts=torch.floor(uniform_draws.size(1)*fraction+0.5).long().cpu()
            self.stats["confidence_sum"]+=confidence.sum().item()
            self.stats["confidence_targets"]+=confidence.numel()
            return counts
