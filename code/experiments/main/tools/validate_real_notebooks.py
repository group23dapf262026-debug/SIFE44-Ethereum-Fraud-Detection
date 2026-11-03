"""Execute delivered literal cells on real inputs; keep reduced GNN smoke scope explicit."""
import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw',required=True);parser.add_argument('--processed-root',required=True)
    parser.add_argument('--ponzi-attributes',required=True)
    parser.add_argument('--datasets',default='Phish,Ponzi');parser.add_argument('--models',default='rf,lightgbm,metaifd,gat,sage')
    parser.add_argument('--output-dir',default='real_notebooks_final')
    args=parser.parse_args();output=ROOT/'validation'/args.output_dir;output.mkdir(exist_ok=True)
    labels=dict(metaifd='MetaIFD',gat='GAT',sage='GraphSAGE',lightgbm='LightGBM',rf='RF')
    for ds in args.datasets.split(','):
        for model in args.models.split(','):
            path=next((ROOT/ds).glob('*_'+labels[model]+'_SIFE20_RUN_ALL.ipynb'))
            work=output/ds/model
            if ds=='Ponzi':
                # Reuse already raw-restored/read-only sidecars without copying1.1GB per notebook.
                # prepare_data still verifies every full-file hash against independent provenance.
                folder=work/'data/float64_attributes';folder.mkdir(parents=True,exist_ok=True)
                for source in Path(args.ponzi_attributes).glob('*.npy'):
                    target=folder/source.name
                    if not target.exists():os.link(source,target)
            history=[];shell=SimpleNamespace(history_manager=SimpleNamespace(input_hist_raw=history))
            namespace={'torch':torch,'get_ipython':lambda:shell,
                       'display':lambda value:print(value.to_string(index=False) if hasattr(value,'to_string') else value)}
            log=output/f'{ds}_{model}_execution.txt'
            with log.open('w',encoding='utf-8') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                print('Actual delivered notebook:',path)
                print('Tabular:full strict roots. GNN:explicit24root balanced reduction before splitting. Mode:smoke,all supported arms,one seed,two GNN epochs.')
                try:
                    for cell in json.loads(path.read_text(encoding='utf-8'))['cells']:
                        if cell['cell_type']!='code':continue
                        source=''.join(cell['source']);tags=cell.get('metadata',{}).get('tags',[])
                        if 'environment' in tags:continue  # Existing tested environment; setup is separately probed.
                        history.append(source);exec(compile(source,str(path),'exec'),namespace)
                        if 'configuration' in tags:
                            namespace.update(RUN_MODE='smoke',DATA_RAW=Path(args.raw),
                                DATA_PHISH=Path(args.processed_root)/ds/'data.pt',OUTPUT_ROOT=output,
                                WORK_ROOT=work,DATA_OUTPUT=work/'data',JOB_DIR=work/'results/smoke',DEVICE='cpu')
                        if 'prepare-data' in tags and model in ('gat','sage','metaifd'):
                            data=namespace['data'];roots=data['roots']
                            take=np.sort(np.r_[np.flatnonzero(roots['labels']==0)[:12],np.flatnonzero(roots['labels']==1)[:12]])
                            data['roots']={key:value[take].copy() for key,value in roots.items()}
                            data['manifest']['validation_only_reduced_roots']=data['roots']['ids'].tolist()
                            print('Explicit validation-only cohort reduction:',len(take),'roots')
                    study=namespace['study'];fits=len(namespace['study_lock']['split_lock_hashes'])*len(study['protocol']['arms'])
                    report=dict(status='PASS',dataset=ds,model=model,fits=fits,scientific_functions_mocked=False,
                        actual_notebook_execution_cells=True,root_count=len(namespace['data']['roots']['ids']),
                        mode='smoke',split_seeds=study['protocol']['split_seeds'],model_seeds=study['protocol']['model_seeds'],
                        arms=study['protocol']['arms'],max_epochs=study['protocol']['max_epochs'],fanouts=study['protocol']['fanouts'],
                        ponzi_sidecars='verified raw-restored float64 read-only hardlinks' if ds=='Ponzi' else None,
                        full_twenty_partition_benchmark_completed=False)
                    (output/f'{ds}_{model}_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                except Exception:
                    traceback.print_exc();raise
            namespace.clear();gc.collect();print(ds,model,'PASS',flush=True)

if __name__=='__main__':main()
