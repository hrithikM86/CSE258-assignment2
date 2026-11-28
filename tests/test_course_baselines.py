"""Hand-calculated model checks and a synthetic full pipeline integration test."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from course_baselines.item_cf import ItemCF,interaction_matrix
from course_baselines.bpr import BPR,TripletSampler
from course_baselines.protocol import load_data,evaluate

ROOT=Path(__file__).resolve().parents[1]


class CourseBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_jaccard_and_cosine_match_hand_computation(self):
        history={1:[1,2],2:[1,3],3:[1,2,4]}
        j=ItemCF.fit(history,5,"jaccard").similarity.toarray()
        c=ItemCF.fit(history,5,"cosine").similarity.toarray()
        self.assertAlmostEqual(j[0,1],2/3,places=6)
        self.assertAlmostEqual(c[0,1],2/np.sqrt(6),places=6)
        self.assertAlmostEqual(j[1,3],.5)
        self.assertAlmostEqual(j[0,3],1/3,places=6)
        self.assertEqual(j[1,2],0)
        np.testing.assert_allclose(j,j.T); np.testing.assert_allclose(c,c.T)
        self.assertFalse(j.diagonal().any()); self.assertFalse(j[4].any())

    def test_cf_scores_full_history_and_ignores_repeat_counts(self):
        h={1:[1,2],2:[1,3],3:[1,2,4]}
        model=ItemCF.fit(h,5)
        scores=model.score_all([1],[[1,2]])[0]
        self.assertAlmostEqual(scores[3],1/3+.5,places=6)
        self.assertAlmostEqual(scores[2],1/3,places=6)
        np.testing.assert_array_equal(model.score_all([1],[[1,2,2]]),model.score_all([1],[[1,2]]))
        self.assertEqual(scores[4],0)

    def test_bpr_negative_exclusion_and_reproducibility(self):
        a=TripletSampler({1:[1,2]},4,42); b=TripletSampler({1:[1,2]},4,42)
        first=a.sample(200); second=b.sample(200)
        for x,y in zip(first,second):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
        users,pos,neg=first
        self.assertEqual(set(pos.tolist()),{1,2})
        self.assertEqual(set(neg.tolist()),{3,4}) # Held-out identities deliberately not excluded.
        with self.assertRaises(ValueError):
            TripletSampler({1:[1,2]},2)

    def test_bpr_loss_and_item_bias(self):
        model=BPR(1,3,2)
        with torch.no_grad():
            model.user_factors.weight.zero_(); model.item_factors.weight.zero_()
            model.item_bias.weight[1]=2
        u=torch.tensor([1]); i=torch.tensor([1]); j=torch.tensor([2])
        torch.testing.assert_close(model.loss(u,i,j,0),F.softplus(torch.tensor(-2.)))
        torch.testing.assert_close(model.loss(u,i,j,.1),F.softplus(torch.tensor(-2.))+.2)
        torch.testing.assert_close(model.score_all([1]),torch.tensor([[2.,0.,0.]]))

    def test_bpr_learns_and_does_not_adapt_to_test_history(self):
        torch.manual_seed(3); model=BPR(1,4,8)
        optimizer=torch.optim.Adam(model.parameters(),lr=.03)
        u=torch.ones(16,dtype=torch.long); i=torch.ones_like(u); j=torch.full_like(u,3)
        initial=model.loss(u,i,j,1e-4).item()
        for _ in range(30):
            optimizer.zero_grad(); loss=model.loss(u,i,j,1e-4); loss.backward(); optimizer.step()
        self.assertLess(model.loss(u,i,j,1e-4).item(),initial*.2)
        torch.testing.assert_close(model.score_all([1],[[1]]),model.score_all([1],[[1,2]]))
        self.assertEqual(model.user_factors.weight[0].abs().sum().item(),0)
        self.assertEqual(model.item_factors.weight[0].abs().sum().item(),0)

    def test_full_catalog_filtering_and_tie_break(self):
        class Scores:
            def score_all(self,users,histories):
                return np.tile(np.array([100.,1.,1.,1.],dtype=np.float32),(len(users),1))
        ranks=evaluate(Scores(),{1:[1]},{1:3})
        self.assertEqual(ranks,{1:2})

    def test_synthetic_pipeline_selects_and_saves_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); split=root/"split"; split.mkdir()
            files={
                "train.csv":(["user_id","item_id","timestamp","rating"],[(1,1,1,5),(1,2,2,4),(2,2,1,4),(2,3,2,5)]),
                "validation.csv":(["user_id","item_id","timestamp","rating"],[(1,3,3,3),(2,4,3,4)]),
                "test.csv":(["user_id","item_id","timestamp","rating"],[(1,4,4,2),(2,5,4,3)]),
                "user_ids.csv":(["reviewer_id","user_id"],[("u1",1),("u2",2)]),
                "item_ids.csv":(["asin","item_id"],[(f"p{i}",i) for i in range(1,7)]),
                "boundary_ties.csv":(["user_id","train_validation_tie","validation_test_tie"],[(1,0,0),(2,0,0)]),
            }
            for name,(header,rows) in files.items():
                with (split/name).open("w",newline="") as file:
                    writer=csv.writer(file); writer.writerow(header); writer.writerows(rows)
            (split/"manifest.json").write_text(json.dumps({"output_sha256":{n:hashlib.sha256((split/n).read_bytes()).hexdigest() for n in files}}))
            training,h,targets,_,_,_=load_data(split,"test")
            self.assertEqual(training[1],[1,2]); self.assertEqual(h[1],[1,2,3]); self.assertEqual(targets[1],4)
            run=root/"run"
            subprocess.run([str(ROOT/".venv/bin/python"),"-m","course_baselines.run","--split-dir",str(split),"--run-dir",str(run),"--epochs","2","--patience","1","--batch-size","2"],cwd=ROOT,check=True,capture_output=True,text=True)
            result=json.loads((run/"results.json").read_text())
            self.assertTrue(result["selection"]["test_not_used_for_selection"])
            self.assertEqual(result["item_cf"]["test"]["all_users"]["users"],2)
            self.assertEqual(len(json.loads((run/"bpr_grid.json").read_text())),4)
            self.assertTrue((run/"bpr.pt").exists()); self.assertTrue((run/"item_cf.npz").exists())
            for model,path in (("bpr",run/"bpr.pt"),("item_cf",run/"item_cf.npz")):
                output=subprocess.run([str(ROOT/".venv/bin/python"),"-m","course_baselines.evaluate","--model",model,"--checkpoint",str(path),"--split-dir",str(split),"--split","test"],cwd=ROOT,check=True,capture_output=True,text=True)
                self.assertEqual(json.loads(output.stdout)["all_users"],result[model]["test"]["all_users"])


if __name__=="__main__":
    unittest.main()
