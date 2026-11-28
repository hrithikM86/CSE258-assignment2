"""Controls, sampler semantics, data integrity, and semantic cache checks."""
import hashlib
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from gsasrec.model import GSASRec as OriginalModel
from gsasrec.training import gsasrec_loss as original_loss
from gsasrec_adaptive.model import GSASRec
from gsasrec_adaptive.neighbors import build_neighbors, load_or_build
from gsasrec_adaptive.sampling import SemanticSampler, AdaptiveSemanticSampler
from gsasrec_adaptive.training import gsasrec_loss, make_batch

ROOT=Path(__file__).resolve().parents[1]


class AdaptiveSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)
        self.table=np.array([[0,0],[2,3],[1,3],[1,2],[0,0]],dtype=np.int32)
        self.flags=np.array([False,True,True,True,False])

    def test_original_files_unchanged(self):
        hashes=json.loads((ROOT/"gsasrec_adaptive/upstream_files.json").read_text())
        for name,expected in hashes.items():
            self.assertEqual(hashlib.sha256((ROOT/"gsasrec"/name).read_bytes()).hexdigest(),expected,name)

    def test_copied_model_and_uniform_loss_are_identical(self):
        original=OriginalModel(8,max_length=5,hidden_size=8,dropout=.3)
        copied=GSASRec(8,max_length=5,hidden_size=8,dropout=.3)
        copied.load_state_dict(original.state_dict())
        x,y=make_batch([[1,2,3],[2,4]],5)
        original.eval(); copied.eval()
        torch.testing.assert_close(original(x),copied(x),rtol=0,atol=0)
        original.train(); copied.train()
        torch.manual_seed(9); a=original_loss(original,x,y,8,4,.5)
        torch.manual_seed(9); b=gsasrec_loss(copied,x,y,8,4,.5)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        a.backward(); b.backward()
        for p,q in zip(original.parameters(),copied.parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

    def test_fixed_budget_and_semantic_membership(self):
        sampler=SemanticSampler(4,self.table,self.flags,.5,42)
        positives=torch.tensor([1,2,3,4]*100)
        negatives=sampler.sample(positives,4)
        self.assertEqual(negatives.shape,(400,4))
        self.assertTrue((negatives>0).all())
        self.assertFalse(negatives.eq(positives[:,None]).any())
        for row,positive in enumerate(positives.tolist()):
            if self.flags[positive]:
                self.assertTrue(set(negatives[row,-2:].tolist()).issubset(set(self.table[positive])))
        self.assertEqual(sampler.stats,{"total_negatives":1600,"requested_semantic":800,
                                        "actual_semantic":600,"fallback_negatives":200})

    def test_missing_text_falls_back_to_same_uniform_draws(self):
        positives=torch.tensor([4]*100)
        zero=SemanticSampler(4,self.table,self.flags,0,42)
        half=SemanticSampler(4,self.table,self.flags,.5,42)
        torch.manual_seed(19); a=zero.sample(positives,4)
        torch.manual_seed(19); b=half.sample(positives,4)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(half.stats["actual_semantic"],0)

    def test_extra_sampler_rng_does_not_change_model_rng(self):
        a=SemanticSampler(4,self.table,self.flags,0,42)
        b=SemanticSampler(4,self.table,self.flags,.5,42)
        positives=torch.tensor([1,2,3])
        torch.manual_seed(3); a.sample(positives,4); after_a=torch.rand(12)
        torch.manual_seed(3); b.sample(positives,4); after_b=torch.rand(12)
        torch.testing.assert_close(after_a,after_b,rtol=0,atol=0)
        c=SemanticSampler(4,self.table,self.flags,.5,42)
        torch.manual_seed(3); first=c.sample(positives,4)
        d=SemanticSampler(4,self.table,self.flags,.5,42)
        torch.manual_seed(3); second=d.sample(positives,4)
        torch.testing.assert_close(first,second,rtol=0,atol=0)

    def test_curriculum_schedule_and_constant_budget(self):
        sampler=AdaptiveSemanticSampler(4,self.table,self.flags,.5,42,mode="curriculum")
        positives=torch.tensor([1,2,3,4])
        for epoch,expected in ((1,0),(3,0),(4,2),(8,8),(13,16),(50,16)):
            sampler.begin_epoch(epoch)
            negatives=sampler.sample(positives,32)
            self.assertEqual(negatives.shape,(4,32))
            self.assertEqual(sampler.stats["requested_semantic"],4*expected)
            self.assertEqual(sampler.stats["actual_semantic"],3*expected)
            self.assertFalse(negatives.eq(positives[:,None]).any())

    def test_confidence_allocates_by_margin_without_gradients(self):
        from torch import nn
        class Probe(nn.Module):
            def __init__(self):
                super().__init__()
                self.output_embeddings=nn.Embedding.from_pretrained(torch.tensor([[0.],[4.],[0.],[0.],[0.]]),freeze=False)
        model=Probe()
        states=torch.tensor([[1.],[-1.],[0.]],requires_grad=True)
        positives=torch.tensor([1,1,1])
        sampler=AdaptiveSemanticSampler(4,self.table,self.flags,.5,42,mode="confidence")
        sampler.begin_epoch(4)
        draws=sampler.sample(positives,4,model=model,states=states)
        self.assertEqual(sampler.stats["requested_semantic"],2)
        self.assertEqual(sampler.stats["confidence_targets"],3)
        self.assertTrue(set(draws[0,-2:].tolist()).issubset({2,3}))
        self.assertIsNone(states.grad)
        self.assertIsNone(model.output_embeddings.weight.grad)
        self.assertEqual(draws.shape,(3,4))

    def test_adaptive_warmup_matches_uniform_loss_and_gradients(self):
        for mode in ("curriculum","confidence"):
            original=OriginalModel(4,max_length=5,hidden_size=8,dropout=.3)
            copied=GSASRec(4,max_length=5,hidden_size=8,dropout=.3)
            copied.load_state_dict(original.state_dict())
            sampler=AdaptiveSemanticSampler(4,self.table,self.flags,.5,42,mode=mode)
            sampler.begin_epoch(1)
            x,y=make_batch([[1,2,3],[2,4]],5)
            torch.manual_seed(11); a=original_loss(original,x,y,4,3,.5)
            torch.manual_seed(11); b=gsasrec_loss(copied,x,y,4,3,.5,sampler)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            a.backward(); b.backward()
            for first,second in zip(original.parameters(),copied.parameters()):
                torch.testing.assert_close(first.grad,second.grad,rtol=0,atol=0)
            self.assertEqual(sampler.stats["actual_semantic"],0)

    def test_invalid_neighbors_and_no_uniform_component_rejected(self):
        with self.assertRaises(ValueError):
            SemanticSampler(4,self.table,self.flags,1,42)
        bad=self.table.copy(); bad[1,0]=1
        with self.assertRaises(ValueError):
            SemanticSampler(4,bad,self.flags,.5,42)

    def test_test_history_and_shared_filtering_use_available_interactions(self):
        from gsasrec_adaptive.evaluate_run import evaluation_data
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            content={
                "item_ids.csv":(["asin","item_id"],[(f"p{i}",i) for i in range(1,7)]),
                "train.csv":(["user_id","item_id","timestamp","rating"],[(1,1,1,5),(1,2,2,4)]),
                "validation.csv":(["user_id","item_id","timestamp","rating"],[(1,3,3,3)]),
                "test.csv":(["user_id","item_id","timestamp","rating"],[(1,4,4,2)]),
                "boundary_ties.csv":(["user_id","train_validation_tie","validation_test_tie"],[(1,0,0)]),
            }
            for name,(header,rows) in content.items():
                with (root/name).open("w",newline="") as file:
                    writer=csv.writer(file); writer.writerow(header); writer.writerows(rows)
            h,v,_,items,_=evaluation_data(root,"validation")
            self.assertEqual(h,{1:[1,2]}); self.assertEqual(v,{1:3})
            h,t,no_ties,items,n=evaluation_data(root,"test")
            self.assertEqual(h,{1:[1,2,3]}); self.assertEqual(t,{1:4})
            self.assertEqual(items,{1,2}); self.assertEqual(no_ties,{1}); self.assertEqual(n,6)

    def test_neighbor_similarity_ties_padding_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/"vectors.npz"
            vectors=np.zeros((6,384),dtype=np.float32)
            vectors[1,0]=1; vectors[2,0]=1; vectors[3,1]=1; vectors[4,1]=-1
            flags=np.array([False,True,True,True,True,False])
            np.savez_compressed(source,embeddings=vectors,has_text=flags)
            before=hashlib.sha256(source.read_bytes()).hexdigest()
            neighbors,has_text=build_neighbors(source,5,2,chunk_size=2)
            self.assertEqual(neighbors[1].tolist(),[2,3]) # 3 precedes 4 when tied.
            self.assertEqual(neighbors[2].tolist(),[1,3])
            self.assertFalse(neighbors[0].any()); self.assertFalse(neighbors[5].any())
            a,_,meta_a=load_or_build(source,root/"cache",5,2)
            b,_,meta_b=load_or_build(source,root/"cache",5,2)
            np.testing.assert_array_equal(a,b)
            self.assertFalse(meta_a["cache_hit"]); self.assertTrue(meta_b["cache_hit"])
            self.assertEqual(before,hashlib.sha256(source.read_bytes()).hexdigest())


if __name__=="__main__":
    unittest.main()
