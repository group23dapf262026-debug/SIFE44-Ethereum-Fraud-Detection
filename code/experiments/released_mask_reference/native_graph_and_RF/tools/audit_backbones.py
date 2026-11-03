"""Local primary-source parity audit; external files are read-only validation evidence."""
import ast
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv,SAGEConv

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baseline.models import SignedLogGAT,InputEncodedSAGE
from baseline.models_meta import HMSL
sys.path.insert(0,str(ROOT/'tests'))
from test_baseline import fixture


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gat-source',required=True)
    parser.add_argument('--sage-source',required=True)
    parser.add_argument('--author-root',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2);report={}
    graph,_,_=fixture();hidden=32
    for model,source_path in [('gat',Path(args.gat_source)),('sage',Path(args.sage_source))]:
        reference_module=load_module('audited_'+model,source_path)
        config=dict(model='gat' if model=='gat' else 'graphsage',hidden=hidden)
        statistics=dict(mean=torch.linspace(-1,1,14).tolist(),scale=torch.linspace(.5,2,14).tolist())
        reference=(reference_module.StudyModel(14,config,statistics) if model=='gat' else
                   reference_module.StudyModel(14,config))
        actual=SignedLogGAT(hidden,statistics) if model=='gat' else InputEncodedSAGE(hidden)
        actual.load_state_dict(reference.state_dict(),strict=True)
        x=torch.cat([graph['EOA'].x,graph['CA'].x]);edges=[]
        offset={'EOA':0,'CA':len(graph['EOA'].x)}
        for et in graph.edge_types:edges.append(graph[et].edge_index+torch.tensor([[offset[et[0]]],[offset[et[2]]]]))
        edge_index=torch.cat(edges,dim=1)
        reference.eval();actual.eval()
        with torch.no_grad():error=float((reference(x,edge_index)-actual(x,edge_index)).abs().max())
        reference.train();actual.train()
        expected=reference(x,edge_index);observed=actual(x,edge_index)
        train_error=float((expected-observed).detach().abs().max())
        assert error==0 and train_error==0
        expected.square().sum().backward();observed.square().sum().backward()
        gradient_error=0.
        for (name,a),(other,b) in zip(reference.named_parameters(),actual.named_parameters()):
            assert name==other
            if a.grad is not None:
                torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
                gradient_error=max(gradient_error,float((a.grad-b.grad).abs().max()))
        report[model]=dict(source=str(source_path),source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            state_keys_exact=True,eval_max_absolute_error=error,train_max_absolute_error=train_error,
            gradient_max_absolute_error=gradient_error,reference='audited newer group backbone; not accepted BN encoder')
    original=Path(args.author_root)
    previous=sys.modules.get('attention_conv')
    sys.modules['attention_conv']=load_module('pinned_author_attention',original/'attention_conv.py')
    author=load_module('pinned_author_model',original/'model.py')
    reference=author.HMSL(8,2,graph,3,'EOA')
    actual=HMSL(8,2,graph,3,'EOA',input_width=14)
    x={nt:torch.randn(3,4,graph[nt].num_nodes,14) for nt in graph.node_types}
    edges={et:graph[et].edge_index for et in graph.edge_types}
    reference.eval();actual.eval()
    # The public implementation has lazy PyG Linear parameters.
    with torch.no_grad():reference(x,edges)
    actual.load_state_dict(reference.state_dict(),strict=True)
    expected,expected_loss=reference(x,edges);observed,observed_loss=actual(x,edges)
    error=float((expected-observed).detach().abs().max());loss_error=float((expected_loss-observed_loss).detach().abs())
    # Different sum kernels can differ at rounding scale; do not promise bitwise cross-device equality.
    torch.testing.assert_close(expected,observed,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(expected_loss,observed_loss,rtol=1e-5,atol=1e-6)
    expected.sum().backward();observed.sum().backward()
    gradient_error=0.
    for (name,a),(other,b) in zip(reference.named_parameters(),actual.named_parameters()):
        assert name==other
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-5)
            gradient_error=max(gradient_error,float((a.grad-b.grad).abs().max()))
    report['metaifd']=dict(source_commit='439c6daa936f67f19486be76642e690a2153d0c5',
        source='reference_author/model.py + attention_conv.py',state_keys_exact=True,
        logits_max_absolute_error=error,loss_absolute_error=loss_error,gradient_max_absolute_error=gradient_error,
        runner_identical=False)
    if previous is None:sys.modules.pop('attention_conv',None)
    else:sys.modules['attention_conv']=previous
    output=ROOT/'validation/backbone_parity.json';output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
