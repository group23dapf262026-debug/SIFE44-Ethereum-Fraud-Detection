"""Check raw input preservation, historical estimator parity and locked TEST behavior."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from baseline.study import input_frame,fixed_configuration,make_estimator,open_study,prepare_role,fit_and_lock,evaluate_locked,verify_model_lock


def fixture():
    rng=np.random.default_rng(12);x=rng.normal(size=(100,14));labels=np.array([0,0,0,1]*25)
    graph={nt:SimpleNamespace(x=torch.tensor(x,dtype=torch.float32)) for nt in ('EOA','CA')}
    membership={role:pd.DataFrame(dict(node_id=pos,label=labels[pos],address=[f'a{i}' for i in pos]))
                for role,pos in [('train',np.arange(60)),('validation',np.arange(60,80)),('test',np.arange(80,100))]}
    return graph,membership


class ReferenceTests(unittest.TestCase):
    def test_input_is_raw14_not_logged(self):
        x=np.arange(-28,28).reshape(4,14)*1.e6
        np.testing.assert_array_equal(input_frame(x,'lr','reference'),x)
        np.testing.assert_array_equal(input_frame(x,'lightgbm','reference'),x)
        with self.assertRaises(ValueError):input_frame(x,'lr','corrected')
        with self.assertRaises(ValueError):input_frame(np.ones((4,58)),'lr','reference')

    def test_lr_matches_historical_pipeline(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(80,14));y=np.array([0]*60+[1]*20)
        reference=make_pipeline(StandardScaler(),LogisticRegression(C=10.,solver='liblinear',max_iter=5000,class_weight=None,random_state=12)).fit(x,y)
        candidate=make_estimator('lr',fixed_configuration('lr','reference'),12,2).fit(input_frame(x,'lr','reference'),y)
        np.testing.assert_allclose(reference.predict_proba(x),candidate.predict_proba(input_frame(x,'lr','reference')),rtol=0,atol=1e-12)
        self.assertIsNone(candidate['classifier'].class_weight)

    def test_lightgbm_keeps_reference_budget(self):
        model=make_estimator('lightgbm',fixed_configuration('lightgbm','reference'),12,2)
        self.assertEqual((model.n_estimators,model.learning_rate,model.num_leaves),(300,.01,31))
        self.assertIsNone(model.class_weight)

    def test_train_scaler_lock_resume_and_tamper(self):
        graph,membership=fixture()
        with tempfile.TemporaryDirectory() as temp:
            study=open_study(graph,membership,{'fixture':True},temp,'Phish','lr','reference',{})
            with self.assertRaisesRegex(ValueError,'all five seeds'):prepare_role(study,'test')
            lock=fit_and_lock(study);self.assertEqual(len(lock['fits']),5)
            estimator=joblib.load(Path(temp)/'fits/seed_12/estimator.joblib')
            expected=graph['EOA'].x[:60].numpy().astype(np.float64).mean(0)
            np.testing.assert_allclose(estimator['scale'].mean_,expected)
            self.assertEqual(estimator['scale'].n_samples_seen_,60)
            first=evaluate_locked(study);second=evaluate_locked(study)
            self.assertEqual(len(first),15);np.testing.assert_allclose(first.macro_f1,second.macro_f1)
            self.assertEqual(fit_and_lock(study),lock)
            path=Path(temp)/'fits/seed_12/estimator.joblib';path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):verify_model_lock(study)


if __name__=='__main__':unittest.main()
