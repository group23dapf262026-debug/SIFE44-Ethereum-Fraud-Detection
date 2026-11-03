"""Scientific integrity checks, including real native sampling and gradient updates."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from baseline.utilities import read_json
from baseline.models import SignedLogGAT,InputEncodedSAGE,materialize_decoder
from baseline.study import (MODELS,fixed_configuration,open_study,prepare_role,fit_and_lock,
    evaluate_locked,verify_model_lock,export_results,classification_metrics,fit_one_seed,fit_gat_statistics)

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    rng=np.random.default_rng(19);graph=HeteroData()
    for nt in ('CA','EOA'):
        graph[nt].x=torch.tensor(rng.normal(size=(24,14)),dtype=torch.float32)
        graph[nt].y=torch.tensor([0,1]*12)
    types=[('CA','call','CA'),('EOA','call','CA'),('EOA','trans','CA'),
           ('CA','trans','CA'),('CA','trans','EOA'),('EOA','trans','EOA')]
    for et in types:
        src=np.repeat(np.arange(24),3);dst=(src+np.tile([0,1,3],24))%24
        graph[et].edge_index=torch.tensor(np.stack([src,dst]),dtype=torch.long)
    membership={}
    for role,ids in dict(train=np.arange(8),validation=np.arange(8,12),test=np.arange(12,16)).items():
        membership[role]=pd.DataFrame(dict(node_id=ids,label=graph['EOA'].y[ids].numpy(),
            address=[f'0x{i:040x}' for i in ids]))
    manifest=dict(split_report=dict(rule='test fixture',sizes={r:len(p) for r,p in membership.items()}))
    return graph,membership,manifest


class BaselineTests(unittest.TestCase):
    def test_twelve_literal_notebooks_match_source_and_preserve_decorators(self):
        notebooks=sorted(ROOT.glob('*/*.ipynb'));self.assertEqual(len(notebooks),12)
        for path in notebooks:
            notebook=json.loads(path.read_text());cells=notebook['cells'];seen={}
            for cell in cells:
                if cell['cell_type']!='code':continue
                source=''.join(cell['source']);tree=ast.parse(source)
                self.assertFalse(cell.get('outputs'))
                if not source.startswith('# BASELINE_SOURCE_CELL: '):continue
                module,key=source.splitlines()[0].split(': ',1)[1].split('.',1)
                if key=='imports':continue
                original=ast.parse((ROOT/'baseline'/(module+'.py')).read_text())
                definition=next(n for n in original.body if getattr(n,'name',None)==key)
                self.assertEqual(ast.dump(tree.body[0],include_attributes=False),ast.dump(definition,include_attributes=False))
                seen[module+'.'+key]=True
            self.assertIn('data.open_raw_member',seen)
            tags=[t for c in cells for t in c.get('metadata',{}).get('tags',[])]
            self.assertLess(tags.index('fit-and-lock'),tags.index('evaluate-test'))
            text='\n'.join(''.join(c['source']) for c in cells)
            self.assertNotIn('base64',text);self.assertNotIn('exec(',text)
            self.assertIn("RUN_MODE='full'",text)

    def test_fixed_settings_and_no_resplit(self):
        self.assertEqual(len(MODELS),6)
        self.assertEqual(fixed_configuration('metaifd'),dict(hidden=128,lr=.001,views=3,contrast_weight=.1))
        self.assertEqual(fixed_configuration('lr')['solver'],'liblinear')
        source=(ROOT/'baseline/study.py').read_text()
        self.assertNotIn('train_test_split',source);self.assertNotIn('search_grid',source)

    def test_all_six_native_pipeline_lock_replay_resume_and_tamper(self):
        graph,membership,manifest=fixture()
        decoder=json.loads((ROOT/'assets/decoder_tensor_values.json').read_text())['Phish']
        for model_name in MODELS:
            with self.subTest(model=model_name),tempfile.TemporaryDirectory() as temporary:
                study=open_study(graph,membership,manifest,temporary,'Phish',model_name,'cpu',
                                 {'test_source':'literal'},decoder if model_name=='metaifd' else None,'smoke')
                with self.assertRaises(ValueError):prepare_role(study,'test')
                lock=fit_and_lock(study)
                self.assertEqual(len(lock['fits']),1)
                self.assertFalse((study['job']/'cache/test').exists())
                fit=lock['fits'][0];self.assertFalse(fit['test_scored']);self.assertEqual(fit['input_width'],14)
                if model_name=='lr':self.assertEqual(fit['detail']['scaler_rows'],8)
                if model_name not in ('lr','rf','lightgbm'):
                    self.assertEqual(fit['stopped_epoch'],2)
                    self.assertTrue((study['job']/'fits/seed_12/last_checkpoint.pt').exists())
                with patch('baseline.study.fit_one_seed',side_effect=AssertionError('No fitting after lock')):
                    result=evaluate_locked(study);fit_and_lock(study)
                self.assertEqual(len(result),3)
                for role in ('train','validation','test'):
                    pred=pd.read_csv(study['job']/f'predictions/seed_12_{role}.csv')
                    recomputed=classification_metrics(pred.label,pred.score,pred.prediction)
                    stored=result.loc[result.role==role].iloc[0]
                    self.assertAlmostEqual(recomputed['macro_f1'],stored.macro_f1,places=12)
                    self.assertEqual(set(pred.node_id),set(membership[role].node_id))
                with patch('baseline.study.tabular_predictions',side_effect=AssertionError('No new inference')), \
                     patch('baseline.study.run_neural_epoch',side_effect=AssertionError('No new inference')):
                    evaluate_locked(study)
                archive=export_results(study,Path(temporary)/'results.zip');self.assertTrue(archive.exists())
                with self.assertRaises(ValueError):
                    open_study(graph,membership,manifest,temporary,'Phish',model_name,'cpu',{'changed':'source'},
                               decoder if model_name=='metaifd' else None,'smoke')
                checkpoint=study['job']/'fits/seed_12'/fit['checkpoint']
                with checkpoint.open('ab') as stream:stream.write(b'changed')
                with self.assertRaises(ValueError):verify_model_lock(study)

    def test_exact_threshold_ties_use_class_zero(self):
        import baseline.study as module
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'roots.npz'
            np.savez(path,source14=np.zeros((2,14)),labels=[0,1],ids=[0,1])
            class Estimator:
                classes_=np.array([0,1])
                def predict_proba(self,x):return np.array([[.5,.5],[.49,.51]])
            _,rows=module.tabular_predictions(Estimator(),{'path':str(path)})
            self.assertEqual(rows.prediction.tolist(),[0,1])

    def test_neural_earlier_best_checkpoint_is_restored_instead_of_last(self):
        graph,membership,manifest=fixture()
        with tempfile.TemporaryDirectory() as temporary:
            study=open_study(graph,membership,manifest,temporary,'Phish','gat','cpu',{'test':'checkpoint'},run_mode='smoke')
            study['input_statistics']=dict(mean=[0.]*14,scale=[1.]*14)
            epochs=[0]
            def controlled_epoch(study,model,records,role,optimizer=None):
                if optimizer is not None:
                    epochs[0]+=1
                    with torch.no_grad():
                        for value in model.parameters():value.fill_(float(epochs[0]))
                    return {'loss':0.},None
                weight=float(next(model.parameters()).flatten()[0])
                return {'macro_f1':.8 if weight==1. else .2,'loss':0.},None
            with patch('baseline.study.run_neural_epoch',side_effect=controlled_epoch):
                fit=fit_one_seed(study,12,[],[])
            self.assertEqual(fit['best_epoch'],1);self.assertEqual(fit['stopped_epoch'],2)
            path=study['job']/'fits/seed_12'
            best=torch.load(path/'checkpoint.pt',weights_only=True)
            last=torch.load(path/'last_checkpoint.pt',weights_only=True)
            self.assertEqual(float(best['state_dict']['conv1.att_src'].flatten()[0]),1.)
            self.assertEqual(float(last['state_dict']['conv1.att_src'].flatten()[0]),2.)

    def test_graph_bytes_verified_before_loading_and_raw_zip_folder_equivalent(self):
        from baseline.data import obtain_graph,open_raw_member
        import zipfile
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);graph=root/'data.pt';graph.write_bytes(b'wrong graph')
            with patch('torch.load',side_effect=AssertionError('Must not deserialize unknown bytes')):
                with self.assertRaises(ValueError):obtain_graph(graph,root/'out','Phish','0'*64)
            member='Meta-IFD data/Phish_unbalance/EOA_labeled.csv';content=b'name\n0xabc\n'
            archive=root/'raw.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr(member,content)
            folder=root/'raw';(folder/'Phish_unbalance').mkdir(parents=True)
            (folder/'Phish_unbalance/EOA_labeled.csv').write_bytes(content)
            for source in (folder,archive):
                with open_raw_member(source,member) as stream:self.assertEqual(stream.read(),content)

    def test_gat_statistics_unique_train_only_and_sage_has_projected_encoder(self):
        from baseline.utilities import atomic_torch_save
        graph,_,_=fixture()
        with tempfile.TemporaryDirectory() as temporary:
            records=[]
            for number,ids in enumerate(([0,1,2],[1,2,3])):
                path=Path(temporary)/f'{number}.pt'
                atomic_torch_save(dict(node_ids={nt:torch.tensor(ids) for nt in graph.node_types}),path)
                records.append(dict(path=str(path)))
            statistics=fit_gat_statistics(graph,records)
            values=torch.cat([graph[nt].x[:4].double() for nt in ('EOA','CA')])
            values=values.sign()*values.abs().log1p();mean=values.mean(0);scale=(values-mean).square().mean(0).sqrt()
            torch.testing.assert_close(torch.tensor(statistics['mean']),mean.float(),rtol=0,atol=0)
            torch.testing.assert_close(torch.tensor(statistics['scale']),scale.float(),rtol=0,atol=0)
            self.assertEqual(statistics['unique_train_nodes'],8)
            before=copy.deepcopy(statistics)
            for nt in graph.node_types:graph[nt].x[4:]*=1000;graph[nt].y[:]=1-graph[nt].y
            self.assertEqual(before,fit_gat_statistics(graph,records))
        model=InputEncodedSAGE(32)
        self.assertTrue(model.conv1.project and model.conv2.project)
        self.assertEqual(model.conv1.aggr,'mean')
        self.assertFalse(any(isinstance(layer,(torch.nn.BatchNorm1d,torch.nn.Dropout)) for layer in model.modules()))
        x=torch.randn(8,14)
        torch.testing.assert_close(model.encode_input(x),torch.tanh(model.input_encoder(x)),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
