"""Scientific controls for paired arms, global test locks and real native GNN training."""
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sife20.features import extract_sife44,select_inputs,PURE36_INDICES
from sife20.models import build_classifier,forward_classifier,ARM_WIDTHS,arm_columns
from sife20.protocol import MODELS,TABULAR,open_study,open_split,supported_arms,verify_split_lock,verify_study_lock,stratified_parts
from sife20.cache import prepare_role,fit_training_statistics
from sife20.training import fit_all,fit_split,fit_arm_seed,neural_epoch,SessionBudgetReached
from sife20.evaluation import evaluate_all,export_results,summarize
from sife20.metrics import classification_metrics
from sife20.utilities import read_json,write_json,json_hash,sha256

ROOT=Path(__file__).resolve().parents[1]


def fixture(dataset='Phish'):
    rng=np.random.default_rng(7);graph=HeteroData();counts={'EOA':24,'CA':24};edges={}
    for nt in ('CA','EOA'):
        graph[nt].x=torch.tensor(rng.normal(size=(24,14)),dtype=torch.float32)
        graph[nt].y=torch.tensor([0,1]*12)
    for et in [('CA','call','CA'),('EOA','call','CA'),('EOA','trans','CA'),
               ('CA','trans','CA'),('CA','trans','EOA'),('EOA','trans','EOA')]:
        source=np.repeat(np.arange(24),3);dest=(source+np.tile([0,1,3],24))%24
        ix=np.stack([source,dest]);attributes=np.column_stack([rng.integers(1,4,len(source)),rng.uniform(0,4,len(source))])
        graph[et].edge_index=torch.tensor(ix,dtype=torch.long)
        edges[et]=dict(edge_index=ix,edge_attr=attributes)
    target='EOA' if dataset=='Phish' else 'CA';ids=np.arange(24)
    result=extract_sife44(counts,edges,target,ids)
    roots=dict(ids=ids,labels=graph[target].y.numpy(),addresses=np.asarray([f'0x{i:040x}' for i in ids]),
               source14=graph[target].x.numpy(),sife44=result['sife44'])
    return dict(graph=graph,edges=edges,roots=roots,node_counts=counts,
                manifest=dict(dataset=dataset,scope='scientific fixture'))


class StudyChecks(unittest.TestCase):
    def test_all_five_train_lock_evaluate_resume_real_native(self):
        decoder=json.loads((ROOT/'assets/decoder_tensor_values.json').read_text())['Phish']
        for model in MODELS:
            with self.subTest(model=model),tempfile.TemporaryDirectory() as temporary:
                study=open_study(fixture(),temporary,'Phish',model,'cpu',{'fixture':'code'},
                                 decoder if model=='metaifd' else None,'smoke')
                split=open_split(study,2001)
                with self.assertRaises(ValueError):prepare_role(study,split,'test')
                lock=fit_all(study)
                split_lock=verify_split_lock(study,split)
                self.assertEqual(len(split_lock['fits']),3 if model=='metaifd' else 4)
                self.assertFalse((split['folder']/'cache/test').exists())
                for fit in split_lock['fits']:
                    self.assertEqual(fit['n_features'],ARM_WIDTHS[fit['arm']]);self.assertFalse(fit['test_scored'])
                    if model=='lr':self.assertEqual(fit['detail']['scaler_fit_rows'],len(split['parts']['train']))
                    if model not in TABULAR:self.assertEqual(fit['stopped_epoch'],2)
                with patch('sife20.training.fit_arm_seed',side_effect=AssertionError('No fit after lock')):
                    metrics=evaluate_all(study);fit_all(study)
                self.assertEqual(len(metrics),len(split_lock['fits'])*3)
                for fit in split_lock['fits']:
                    predictions=pd.read_csv(study['output']/fit['trial']/'predictions.csv')
                    for role,frame in predictions.groupby('role'):
                        recomputed=classification_metrics(frame.label,frame.score,frame.prediction)
                        stored=metrics.loc[(metrics.arm==fit['arm'])&(metrics.role==role)].iloc[0]
                        self.assertAlmostEqual(recomputed['macro_f1'],stored.macro_f1,places=12)
                        self.assertEqual(set(frame.node_id),set(split['membership'][role].node_id))
                with patch('sife20.evaluation.tabular_predict',side_effect=AssertionError('No new inference')), \
                     patch('sife20.evaluation.neural_epoch',side_effect=AssertionError('No new inference')):
                    evaluate_all(study)
                archive=export_results(study,Path(temporary)/'results.zip');self.assertTrue(archive.exists())
                checkpoint=study['output']/split_lock['fits'][0]['trial']/split_lock['fits'][0]['checkpoint']
                with checkpoint.open('ab') as stream:stream.write(b'tamper')
                with self.assertRaises(ValueError):verify_study_lock(study)

    def test_pure36_neural_does_not_read_source14(self):
        for model in ('gat','sage'):
            with self.subTest(model=model),tempfile.TemporaryDirectory() as temporary:
                study=open_study(fixture(),temporary,'Phish',model,'cpu',{'fixture':'pure'},run_mode='smoke')
                split=open_split(study,2001);records=prepare_role(study,split,'train')
                stats=fit_training_statistics(study,split)
                classifier=build_classifier(model,'Pure36',study['protocol']['config'],study['data']['graph'],'EOA',stats)
                classifier.eval();batch=torch.load(records[0]['path'],weights_only=True)
                batch['batch_name']=Path(records[0]['path']).stem
                with torch.no_grad():expected,_=forward_classifier(classifier,study,split,'Pure36',batch,'train')
                # NaNs prove the Pure36 forward path does not consume source14 at all.
                for nt in study['data']['graph'].node_types:study['data']['graph'][nt].x[:]=float('nan')
                with torch.no_grad():observed,_=forward_classifier(classifier,study,split,'Pure36',batch,'train')
                torch.testing.assert_close(expected,observed,rtol=0,atol=0)
                self.assertTrue(bool(torch.isfinite(observed).all()))

    def test_meta_extra_projection_equals_explicit_concatenation(self):
        data=fixture();graph=data['graph'];statistics=dict(mean=[0.]*58,scale=[1.]*58)
        for arm,extra_width in [('X14_Pure36',36),('X14_SIFE44',44)]:
            model=build_classifier('metaifd',arm,dict(hidden=8,views=3),graph,'EOA',statistics)
            model.eval();reference=copy.deepcopy(model)
            views={nt:torch.randn(3,4,24,14) for nt in graph.node_types}
            extra={nt:torch.randn(24,extra_width) for nt in graph.node_types}
            expanded={nt:torch.cat([value,extra[nt][None,None].expand(3,4,-1,-1)],dim=-1) for nt,value in views.items()}
            edges={et:graph[et].edge_index for et in graph.edge_types}
            with torch.no_grad():
                actual,loss=model(views,edges,extra)
                expected,reference_loss=reference(expanded,edges)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(loss,reference_loss,rtol=1e-5,atol=1e-6)

    def test_same_membership_across_models_and_seed_pairs(self):
        data=fixture();reference=None
        for model in MODELS:
            with tempfile.TemporaryDirectory() as temporary:
                study=open_study(data,temporary,'Phish',model,'cpu',{'fixture':'shared'},run_mode='smoke')
                split=open_split(study,2001)
                current={r:f.to_dict('records') for r,f in split['membership'].items()}
                if reference is None:reference=current
                else:self.assertEqual(reference,current)
        for seed in (2001,2002,2020):
            parts=stratified_parts(data['roots'],seed)
            self.assertEqual(len(set(np.concatenate(list(parts.values())))),24)
        self.assertNotEqual(stratified_parts(data['roots'],2001)['test'].tolist(),stratified_parts(data['roots'],2002)['test'].tolist())

    def test_full_budget_and_meta_missing_arm_are_explicit(self):
        counts=0
        for model in MODELS:
            with tempfile.TemporaryDirectory() as temporary:
                study=open_study(fixture(),temporary,'Phish',model,'cpu',{'fixture':'budget'})
                p=study['protocol'];self.assertEqual(p['split_seeds'],list(range(2001,2021)))
                self.assertEqual(p['model_seeds'],list(range(12,17)))
                counts+=20*5*len(p['arms'])
                self.assertFalse(p['hyperparameter_search'])
                if model=='metaifd':self.assertIn('Pure36',p['unsupported_arms'])
        self.assertEqual(counts*2,3800)
        with self.assertRaises(ValueError):
            build_classifier('metaifd','Pure36',{'hidden':8,'views':3},fixture()['graph'],'EOA',{})

    def test_session_budget_does_not_create_test_or_completed_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            study=open_study(fixture(),temporary,'Phish','lightgbm','cpu',{'fixture':'partial'},run_mode='smoke')
            with self.assertRaises(SessionBudgetReached):fit_all(study,deadline=0)
            self.assertFalse((study['output']/'study_lock.json').exists())
            self.assertFalse((study['output']/'split_2001/cache/test').exists())
            partial=export_results(study,Path(temporary)/'partial.zip',allow_partial=True)
            self.assertTrue(partial.exists())
            with zipfile.ZipFile(partial) as archive:
                self.assertFalse(any(name.endswith('/partial.zip') for name in archive.namelist()))
            with self.assertRaises(ValueError):evaluate_all(study)

    def test_second_partition_is_required_before_first_partition_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            study=open_study(fixture(),temporary,'Phish','lightgbm','cpu',{'fixture':'two_partitions'},run_mode='smoke')
            # A labelled test-only fixture adds a second requested split; no user run is modified.
            study['protocol']['split_seeds']=[2001,2002]
            identity=read_json(study['output']/'protocol.json');identity['protocol']=study['protocol']
            write_json(study['output']/'protocol.json',identity);study['identity']=json_hash(identity)
            first=open_split(study,2001);fit_split(study,first)
            with self.assertRaises(ValueError):prepare_role(study,first,'test')
            self.assertFalse((study['output']/'split_2002/lock.json').exists())
            fit_all(study)
            self.assertTrue((study['output']/'split_2002/lock.json').exists())
            records=prepare_role(study,first,'test');self.assertTrue(records)

    def test_earlier_validation_checkpoint_is_restored(self):
        with tempfile.TemporaryDirectory() as temporary:
            study=open_study(fixture(),temporary,'Phish','gat','cpu',{'fixture':'earlier_best'},run_mode='smoke')
            split=open_split(study,2001);records=prepare_role(study,split,'train');fit_training_statistics(study,split)
            validation=prepare_role(study,split,'validation');steps=[0]
            def controlled_epoch(study,split,model,records,role,arm,optimizer=None):
                if optimizer is not None:
                    steps[0]+=1
                    with torch.no_grad():
                        for parameter in model.parameters():parameter.fill_(float(steps[0]))
                    return {'loss':0.},None
                weight=float(next(model.parameters()).flatten()[0])
                return {'loss':0.,'macro_f1':.8 if weight==1. else .2},None
            with patch('sife20.training.neural_epoch',side_effect=controlled_epoch):
                fit=fit_arm_seed(study,split,'X14',12,records,validation)
            self.assertEqual(fit['best_epoch'],1);self.assertEqual(fit['stopped_epoch'],2)
            trial=study['output']/fit['trial']
            best=torch.load(trial/'checkpoint.pt',weights_only=True);last=torch.load(trial/'last_checkpoint.pt',weights_only=True)
            self.assertEqual(float(best['state_dict']['conv1.att_src'].flatten()[0]),1.)
            self.assertEqual(float(last['state_dict']['conv1.att_src'].flatten()[0]),2.)

    def test_statistical_summary_averages_seeds_before_split_sd(self):
        with tempfile.TemporaryDirectory() as temporary:
            study=open_study(fixture(),temporary,'Phish','lightgbm','cpu',{'fixture':'summary'},run_mode='smoke')
            study['protocol']['split_seeds']=[2001,2002];study['protocol']['model_seeds']=[12,13]
            rows=[]
            for split in [2001,2002]:
                for seed in [12,13]:
                    for arm in study['protocol']['arms']:
                        value=.5+(.1 if split==2002 else 0)+(.2 if seed==13 else 0)+(.02 if arm!='X14' else 0)
                        metrics={name:value for name in ['macro_f1','micro_f1','binary_f1','precision','recall','accuracy','average_precision','roc_auc','fpr']}
                        rows.append(dict(dataset='Phish',model='lightgbm',split_seed=split,model_seed=seed,arm=arm,role='test',**metrics))
            result=summarize(study,pd.DataFrame(rows))
            baseline=result.loc[(result.arm=='X14')&(result.metric=='macro_f1')].iloc[0]
            self.assertAlmostEqual(baseline.mean_percent,65.)
            self.assertAlmostEqual(baseline.sd_across_split_means_pp,np.std([60.,70.],ddof=1))
            deltas=pd.read_csv(study['output']/'paired_macro_f1_deltas_pp.csv')
            np.testing.assert_allclose(deltas.drop(columns='split_seed'),2.)

    def test_literal_notebook_definitions_match_source_including_decorators(self):
        paths=list(ROOT.glob('*/*.ipynb'));self.assertEqual(len(paths),10)
        for path in paths:
            notebook=json.loads(path.read_text())
            for cell in notebook['cells']:
                if cell['cell_type']!='code':continue
                source=''.join(cell['source']);tree=ast.parse(source)
                self.assertFalse(cell.get('outputs'))
                if not source.startswith('# SIFE20_SOURCE_CELL: '):continue
                module,key=source.splitlines()[0].split(': ',1)[1].split('.',1)
                if key=='imports':continue
                original=ast.parse((ROOT/'sife20'/(module+'.py')).read_text(encoding='utf-8'))
                expected=next(n for n in original.body if getattr(n,'name',None)==key)
                self.assertEqual(ast.dump(expected,include_attributes=False),ast.dump(tree.body[0],include_attributes=False))
            text='\n'.join(''.join(c['source']) for c in notebook['cells'])
            self.assertNotIn('base64',text);self.assertNotIn('exec(',text)
            self.assertIn('probe_native_current_kernel',text) if any(s in path.name for s in ['GAT','GraphSAGE','MetaIFD']) else None

    def test_actual_literal_execution_cells_on_toy_lightgbm_data(self):
        """Catch missing notebook-global dependencies that normal package imports can hide."""
        path=next((ROOT/'Phish').glob('*_LightGBM_SIFE20_RUN_ALL.ipynb'))
        history=[];shell=SimpleNamespace(history_manager=SimpleNamespace(input_hist_raw=history))
        namespace={'torch':torch,'get_ipython':lambda:shell,'display':lambda value:None}
        with tempfile.TemporaryDirectory() as temporary:
            for cell in json.loads(path.read_text())['cells']:
                if cell['cell_type']!='code':continue
                source=''.join(cell['source']);tags=cell.get('metadata',{}).get('tags',[])
                if 'environment' in tags:continue
                history.append(source)
                if 'prepare-data' in tags:
                    # This check intentionally substitutes a documented fixture for raw preparation.
                    namespace['data']=fixture();namespace['SOURCE_SIGNATURE']=namespace['notebook_source_signature']()
                    continue
                exec(compile(source,str(path),'exec'),namespace)
                if 'configuration' in tags:
                    output=Path(temporary)
                    namespace.update(RUN_MODE='smoke',OUTPUT_ROOT=output,WORK_ROOT=output/'work',
                                     DATA_OUTPUT=output/'data',JOB_DIR=output/'results/smoke',DEVICE='cpu')
            self.assertTrue((Path(temporary)/'Phish_lightgbm_sife20_five_smoke_results.zip').exists())
            self.assertEqual(len(namespace['metrics']),12)


if __name__=='__main__':unittest.main()
