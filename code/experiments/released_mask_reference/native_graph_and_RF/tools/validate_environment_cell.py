"""Execute the literal delivered GNN environment cell and record all six setup hashes."""
import contextlib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    sources={};hashes={}
    for dataset in ('Phish','Ponzi'):
        for model in ('MetaIFD','GAT','GraphSAGE'):
            path=next((ROOT/dataset).glob('*_'+model+'_BASELINE_RUN_ALL.ipynb'))
            notebook=json.loads(path.read_text(encoding='utf-8'))
            source=next(''.join(c['source']) for c in notebook['cells'] if 'environment' in c.get('metadata',{}).get('tags',[]))
            digest=hashlib.sha256(source.encode()).hexdigest()
            hashes[path.relative_to(ROOT).as_posix()]=digest;sources[digest]=source
    assert len(sources)==1,'All six GNN notebooks should share one environment cell'
    source=next(iter(sources.values()))
    (ROOT/'KAGGLE_GNN_ENVIRONMENT_FIX.py').write_text(source,encoding='utf-8')
    namespace={}
    output=ROOT/'validation/environment_cell_execution.txt'
    with output.open('w',encoding='utf-8') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        # This is the actual visible environment cell, with no function mocks.
        exec(compile(source,'delivered_gnn_environment_cell','exec'),namespace)
    report=dict(status='PASS',shared_environment_cell_notebooks=hashes,
        actual_cell_executed=True,functions_mocked=False,real_native_probe=namespace['NATIVE_ENVIRONMENT_REPORT'],
        installation_branch='simulated regression test; installed-backend probe actually executed',
        kaggle_cuda_execution_verified=False,scientific_configuration_changed=False)
    (ROOT/'validation/environment_cell_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
