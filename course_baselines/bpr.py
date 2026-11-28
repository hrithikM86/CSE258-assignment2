"""BPR matrix factorization with explicit pairwise logistic optimization."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TripletSampler:
    """Bootstrap observed positive pairs, uniform negatives outside training history."""
    def __init__(self,histories,num_items,seed=42):
        self.num_items=num_items
        self.users=np.array([u for u,h in histories.items() for _ in h],dtype=np.int64)
        self.positives=np.array([i for h in histories.values() for i in h],dtype=np.int64)
        if not len(self.users):
            raise ValueError("No training interactions.")
        if any(len(set(h))>=num_items for h in histories.values()):
            raise ValueError("A training user has no eligible negatives.")
        self.known=np.sort(self.users*(num_items+1)+self.positives)
        self.rng=np.random.default_rng(seed)

    def _known(self,users,items):
        codes=users*(self.num_items+1)+items
        positions=np.searchsorted(self.known,codes)
        return (positions<len(self.known)) & (self.known[np.minimum(positions,len(self.known)-1)]==codes)

    def sample(self,size):
        if size<1:
            raise ValueError("Batch size must be positive.")
        index=self.rng.integers(len(self.users),size=size)
        users=self.users[index]; positives=self.positives[index]
        negatives=self.rng.integers(1,self.num_items+1,size=size)
        rejected=self._known(users,negatives)
        while rejected.any():
            negatives[rejected]=self.rng.integers(1,self.num_items+1,size=rejected.sum())
            rejected=self._known(users,negatives)
        return tuple(torch.from_numpy(a) for a in (users,positives,negatives))


class BPR(nn.Module):
    def __init__(self,num_users,num_items,factors=32):
        super().__init__()
        if min(num_users,num_items,factors)<1:
            raise ValueError("Model sizes must be positive.")
        self.user_factors=nn.Embedding(num_users+1,factors,padding_idx=0)
        self.item_factors=nn.Embedding(num_items+1,factors,padding_idx=0)
        self.item_bias=nn.Embedding(num_items+1,1,padding_idx=0)
        nn.init.normal_(self.user_factors.weight,std=.01)
        nn.init.normal_(self.item_factors.weight,std=.01)
        nn.init.zeros_(self.item_bias.weight)
        with torch.no_grad():
            self.user_factors.weight[0].zero_(); self.item_factors.weight[0].zero_()

    def loss(self,users,positives,negatives,regularization=1e-4):
        if regularization<0:
            raise ValueError("Regularization must be nonnegative.")
        user=self.user_factors(users)
        positive=self.item_factors(positives); negative=self.item_factors(negatives)
        bp=self.item_bias(positives).squeeze(-1); bn=self.item_bias(negatives).squeeze(-1)
        margin=(user*(positive-negative)).sum(-1)+bp-bn
        ranking=F.softplus(-margin).mean()
        l2=(user.square().sum(-1)+positive.square().sum(-1)+negative.square().sum(-1)+bp.square()+bn.square()).mean()/2
        return ranking+regularization*l2

    def score_all(self,users,histories=None):
        ids=torch.as_tensor(users,dtype=torch.long,device=self.user_factors.weight.device)
        return self.user_factors(ids)@self.item_factors.weight[1:].T+self.item_bias.weight[1:,0]
