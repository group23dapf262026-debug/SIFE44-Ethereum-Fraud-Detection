"""Verify that X14 matches the corrected native reference backbones before feature extension."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from sife20.models import StudyGAT,StudySAGE,build_classifier
from test_study import fixture


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference-root',required=True);args=parser.parse_args()
    reference_root=Path(args.reference_root);sys.path.insert(0,str(reference_root))
    from baseline.models import SignedLogGAT,InputEncodedSAGE
    from baseline.models_meta import HMSL as ReferenceHMSL
    torch.set_num_threads(2);data=fixture();graph=data['graph'];report={}
    mean=torch.linspace(-1.,1.,58);scale=torch.linspace(.5,2.,58)
    stats=dict(mean=mean.tolist(),scale=scale.tolist());small=dict(mean=mean[:14].tolist(),scale=scale[:14].tolist())
    x=torch.cat([graph['EOA'].x,graph['CA'].x]);offset={'EOA':0,'CA':24}
    edges=torch.cat([graph[et].edge_index+torch.tensor([[offset[et[0]]],[offset[et[2]]]]) for et in graph.edge_types],dim=1)
    for model,reference,actual in [('gat',SignedLogGAT(32,small),StudyGAT('X14',32,stats)),
                                   ('sage',InputEncodedSAGE(32),StudySAGE('X14',32,stats))]:
        differences=actual.load_state_dict(reference.state_dict(),strict=False)
        assert not differences.unexpected_keys
        assert set(differences.missing_keys)==({'input_mean','input_scale'} if model=='sage' else set())
        assert sum(v.numel() for v in reference.parameters())==sum(v.numel() for v in actual.parameters())
        reference.eval();actual.eval();expected=reference(x,edges);observed=actual(x,edges)
        torch.testing.assert_close(expected,observed,rtol=0,atol=0)
        expected.square().sum().backward();observed.square().sum().backward()
        for (name,a),(other,b) in zip(reference.named_parameters(),actual.named_parameters()):
            assert name==other
            if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
        report[model]=dict(X14_logits_max_error=float((expected-observed).detach().abs().max()),
                           parameter_count=sum(v.numel() for v in actual.parameters()),gradients_exact=True)
    reference=ReferenceHMSL(8,2,graph,3,'EOA',input_width=14)
    actual=build_classifier('metaifd','X14',dict(hidden=8,views=3),graph,'EOA',stats)
    actual.load_state_dict(reference.state_dict(),strict=True)
    views={nt:torch.randn(3,4,graph[nt].num_nodes,14) for nt in graph.node_types}
    typed_edges={et:graph[et].edge_index for et in graph.edge_types}
    expected,reference_loss=reference(views,typed_edges);observed,actual_loss=actual(views,typed_edges)
    torch.testing.assert_close(expected,observed,rtol=0,atol=0);torch.testing.assert_close(reference_loss,actual_loss,rtol=0,atol=0)
    report['metaifd']=dict(X14_logits_max_error=float((expected-observed).detach().abs().max()),
                          contrastive_loss_error=float((reference_loss-actual_loss).detach().abs()))
    report['reference_source_hashes']={name:hashlib.sha256((reference_root/'baseline'/name).read_bytes()).hexdigest()
                                     for name in ['models.py','models_meta.py']}
    output=ROOT/'validation/backbone_parity.json';output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
