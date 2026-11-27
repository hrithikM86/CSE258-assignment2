"""Behavioral and independent numerical checks for the SASRec reproduction."""
import math
import unittest
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gsasrec.evaluate import rank_targets
from sasrec.data import TrainingSampler, left_pad
from sasrec.evaluate import evaluate, summarize
from sasrec.model import SASRec
from sasrec.training import sasrec_loss


def reference_forward(model, ids):
    """NumPy transcription of the reference TF operations (dropout disabled)."""
    weights = {k: v.detach().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    def norm(x, prefix):
        return ((x-x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True)+1e-8)
                * weights[prefix+".weight"] + weights[prefix+".bias"])
    def linear(x, prefix):
        return x @ weights[prefix+".weight"].T + weights[prefix+".bias"]
    mask = (ids != 0)[..., None]
    x = (weights["item_embeddings.weight"][ids] * math.sqrt(model.hidden_size)
         + weights["position_embeddings.weight"][np.arange(model.max_length)]) * mask
    for index, block in enumerate(model.blocks):
        prefix = f"blocks.{index}"
        query = norm(x, prefix+".attention_norm")
        heads = block.num_heads
        def split(t):
            return t.reshape(t.shape[0], t.shape[1], heads, -1).transpose(0,2,1,3)
        q, k, v = (split(linear(t, prefix+"."+name))
                   for t, name in ((query,"query"),(x,"key"),(x,"value")))
        logits = q @ k.transpose(0,1,3,2) / math.sqrt(block.head_size)
        key_valid = np.abs(x).sum(-1) != 0
        allowed = np.tril(np.ones((model.max_length, model.max_length), dtype=bool))[None,None] & key_valid[:,None,None,:]
        logits = np.where(allowed, logits, -(2**32)+1)
        a = np.exp(logits-logits.max(-1,keepdims=True))
        a = a / a.sum(-1,keepdims=True)
        a *= (np.abs(query).sum(-1)!=0)[:,None,:,None]
        attended = (a @ v).transpose(0,2,1,3).reshape(x.shape)
        y = norm(query+attended, prefix+".feedforward_norm")
        x = (y+linear(np.maximum(linear(y,prefix+".feedforward_in"),0),prefix+".feedforward_out"))*mask
    return norm(x,"final_norm")


class SASRecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)

    def test_original_operations_against_numpy(self):
        for heads in (1, 2):
            with self.subTest(heads=heads):
                model = SASRec(10, max_length=5, hidden_size=8, num_heads=heads, dropout=0).eval()
                # Exercise learned biases, including normalization beta at padding.
                with torch.no_grad():
                    for name, value in model.named_parameters():
                        if name.endswith("bias"):
                            value.uniform_(-0.1, 0.1)
                ids = torch.tensor([[0,0,1,2,3],[0,4,5,6,7],[0,0,0,0,0]])
                actual = model(ids).detach().numpy()
                np.testing.assert_allclose(actual, reference_forward(model,ids.numpy()), atol=2e-5, rtol=2e-5)

    def test_future_items_cannot_change_past_states(self):
        model = SASRec(10, max_length=5, hidden_size=8, dropout=0).eval()
        a = model(torch.tensor([[0,1,2,3,4]]))
        b = model(torch.tensor([[0,1,2,8,9]]))
        torch.testing.assert_close(a[:,:3], b[:,:3], atol=1e-6, rtol=1e-6)

    def test_padding_is_finite_and_batch_independent(self):
        model = SASRec(10, max_length=5, hidden_size=8, dropout=0).eval()
        ids = torch.tensor([[0,0,0,0,0],[0,0,0,0,1],[1,2,3,4,5]])
        states = model(ids)
        self.assertTrue(torch.isfinite(states).all())
        torch.testing.assert_close(states[1],model(ids[1:2])[0])
        with self.assertRaises(ValueError):
            model(ids[:,:4])

    def test_sampler_alignment_and_training_only_exclusion(self):
        histories = {1: [1,2,3,4]}
        x, y, z = TrainingSampler(histories,6,3,42).sample(128)
        self.assertTrue(torch.equal(x,torch.tensor([[1,2,3]]).expand_as(x)))
        self.assertTrue(torch.equal(y,torch.tensor([[2,3,4]]).expand_as(y)))
        self.assertEqual(set(z.flatten().tolist()),{5,6})
        # Item 5 could be held out: it must remain eligible, preventing leakage.
        torch.testing.assert_close(left_pad([[1,2],[1,2,3,4]],3),torch.tensor([[0,1,2],[2,3,4]]))
        x,y,z = TrainingSampler({1:[1,2]},6,4,42).sample(1)
        self.assertEqual(x.tolist(),[[0,0,0,1]])
        self.assertEqual(y.tolist(),[[0,0,0,2]])
        self.assertTrue(torch.equal(z.eq(0),y.eq(0)))

    def test_sampler_reproducibility_and_no_negative_guard(self):
        a = TrainingSampler({1:[1,2],2:[3,4]},6,4,42)
        b = TrainingSampler({1:[1,2],2:[3,4]},6,4,42)
        for x,y in zip(a.sample(8),b.sample(8)):
            torch.testing.assert_close(x,y)
        with self.assertRaises(ValueError):
            TrainingSampler({1:[1,2]},2)

    def test_bce_masking_and_embedding_regularization(self):
        model = SASRec(8,max_length=4,hidden_size=8,dropout=0)
        x,y,z = TrainingSampler({1:[1,2,3]},8,4).sample(2)
        states = model(x); valid = y.ne(0)
        pos = (states[valid]*model.item_embeddings(y[valid])).sum(-1)
        neg = (states[valid]*model.item_embeddings(z[valid])).sum(-1)
        expected = F.binary_cross_entropy_with_logits(pos,torch.ones_like(pos)) + F.binary_cross_entropy_with_logits(neg,torch.zeros_like(neg))
        torch.testing.assert_close(sasrec_loss(model,x,y,z),expected)
        reg = 0.5 * 0.01 * (model.item_embeddings.weight[1:].square().sum()+model.position_embeddings.weight.square().sum())
        torch.testing.assert_close(sasrec_loss(model,x,y,z,0.01),expected+reg)

    def test_training_reduces_loss_and_keeps_padding_zero(self):
        model = SASRec(8,max_length=4,hidden_size=8,dropout=0)
        batch = TrainingSampler({1:[1,2,3]},8,4).sample(8)
        optimizer = torch.optim.Adam(model.parameters(),lr=0.01,betas=(0.9,0.98))
        before = sasrec_loss(model,*batch).item()
        for _ in range(30):
            optimizer.zero_grad()
            loss = sasrec_loss(model,*batch)
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
        self.assertLess(sasrec_loss(model,*batch).item(),before*0.5)
        self.assertEqual(model.item_embeddings.weight[0].abs().sum().item(),0)
        self.assertEqual(model.item_embeddings.weight.grad[0].abs().sum().item(),0)
        self.assertFalse(hasattr(model,"output_embeddings"))

    def test_shared_ranking_and_full_history_filtering(self):
        class FixedModel(nn.Module):
            max_length = 2
            def __init__(self):
                super().__init__()
                self.item_embeddings = nn.Embedding.from_pretrained(torch.tensor([[0.],[100.],[3.],[2.],[1.]]))
                self.inputs = []
            def forward(self,x):
                self.inputs.append(x.clone())
                return torch.ones(*x.shape,1)
        model = FixedModel()
        ranks = evaluate(model,{1:[1,2,3]},{1:4},1)
        self.assertEqual(ranks,{1:1}) # Old item 1 must be filtered beyond truncation.
        self.assertEqual(model.inputs[0].tolist(),[[2,3]])
        tied = rank_targets(torch.tensor([[1.,1.,1.,1.]]),[[4]],[3])
        self.assertEqual(tied.item(),3)
        summary = summarize(ranks,{1},{1:4},{1,2,3})
        self.assertEqual(summary["all_users"]["recall_at_10"],1)
        self.assertEqual(summary["targets_unseen_in_training"],1)

    def test_state_dict_round_trip(self):
        model = SASRec(8,max_length=4,hidden_size=8,dropout=0).eval()
        clone = SASRec(8,max_length=4,hidden_size=8,dropout=0).eval()
        clone.load_state_dict(model.state_dict())
        ids = left_pad([[1,2]],4)
        torch.testing.assert_close(model(ids),clone(ids))


if __name__ == "__main__":
    unittest.main()
