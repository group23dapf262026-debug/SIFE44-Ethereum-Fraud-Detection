"""Check full-control parity with the completed five-model main suite."""
import ast
import importlib
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from test_study import fixture
from sife20.models import StudyGAT,StudySAGE,build_classifier
from sife20.protocol import fixed_configuration

def main():
    reference=ROOT.parent/'SIFE20_FIVE_MODELS_20261028/sife20'
    spec=importlib.util.spec_from_file_location('completed_sife20',reference/'__init__.py',submodule_search_locations=[str(reference)])
    package=importlib.util.module_from_spec(spec);sys.modules['completed_sife20']=package;spec.loader.exec_module(package)
    old=importlib.import_module('completed_sife20.models')
    oldprotocol=importlib.import_module('completed_sife20.protocol')
    torch.set_num_threads(2);torch.manual_seed(12);data=fixture();graph=data['graph']
    stats=dict(mean=torch.linspace(-1,1,58).tolist(),scale=torch.linspace(.5,2,58).tolist())
    x=torch.randn(48,58);offset={'EOA':0,'CA':24}
    edges=torch.cat([graph[et].edge_index+torch.tensor([[offset[et[0]]],[offset[et[2]]]]) for et in graph.edge_types],dim=1)
    report={}
    for name,newclass,oldclass in [('gat',StudyGAT,old.StudyGAT),('sage',StudySAGE,old.StudySAGE)]:
        new=newclass('X14_SIFE44',32,stats);previous=oldclass('X14_SIFE44',32,stats)
        new.load_state_dict(previous.state_dict(),strict=True);new.eval();previous.eval()
        actual=new(x,edges);expected=previous(x,edges)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        actual.square().sum().backward();expected.square().sum().backward()
        for (a,p),(b,q) in zip(new.named_parameters(),previous.named_parameters()):
            assert a==b
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
        report[name]=dict(full58_logits_max_error=float((actual-expected).detach().abs().max()),gradients_exact=True)
    new=build_classifier('metaifd','X14_SIFE44',dict(hidden=8,views=3),graph,'EOA',stats)
    previous=old.build_classifier('metaifd','X14_SIFE44',dict(hidden=8,views=3),graph,'EOA',stats)
    new.load_state_dict(previous.state_dict(),strict=True);new.eval();previous.eval()
    views={nt:torch.randn(3,4,24,14) for nt in graph.node_types};extras={nt:torch.randn(24,44) for nt in graph.node_types}
    typed_edges={et:graph[et].edge_index for et in graph.edge_types}
    a,loss=new(views,typed_edges,extras);b,oldloss=previous(views,typed_edges,extras)
    torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(loss,oldloss,rtol=0,atol=0)
    report['metaifd']=dict(full58_logits_max_error=float((a-b).detach().abs().max()),contrastive_loss_max_error=float((loss-oldloss).detach().abs()))
    unchanged=['data.py','sampling.py','views.py','models_meta.py','attention_conv.py','cache.py','training.py','utilities.py','metrics.py']
    hashes={}
    for name in unchanged:
        value=hashlib.sha256((ROOT/'sife20'/name).read_bytes()).hexdigest()
        assert value==hashlib.sha256((reference/name).read_bytes()).hexdigest(),name
        hashes[name]=value
    def node(path,name):
        return ast.dump(next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if getattr(n,'name',None)==name),include_attributes=False)
    for name in ['safe_ratio','summarize_channel','extract_sife44']:
        assert node(ROOT/'sife20/features.py',name)==node(reference/'features.py',name)
    for name in ['metaifd','gat','sage','lightgbm','rf']:assert fixed_configuration(name)==oldprotocol.fixed_configuration(name)
    report.update(status='PASS',fixed_configs_unchanged=True,unchanged_scientific_modules=hashes,numeric_feature_functions_ast_identical=True,
                  scope='full58neural logits/gradients on fixture,unchanged sampling/training/extraction functions; not full5000fit result parity')
    (ROOT/'validation/completed_control_parity.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
