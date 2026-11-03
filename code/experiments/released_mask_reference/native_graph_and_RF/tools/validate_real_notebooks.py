"""Execute delivered notebook cells on local real inputs; never claim full neural training."""
import argparse
import contextlib
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import traceback

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',default='lr,rf,lightgbm')
    parser.add_argument('--datasets',default='Phish,Ponzi')
    parser.add_argument('--raw',required=True)
    parser.add_argument('--processed-root',required=True)
    parser.add_argument('--output-dir',default='real_notebooks_final')
    args=parser.parse_args()
    output=ROOT/'validation'/args.output_dir;output.mkdir(parents=True,exist_ok=True)
    mapping=dict(metaifd='MetaIFD',gat='GAT',sage='GraphSAGE',lightgbm='LightGBM',lr='LR',rf='RF')
    for dataset in args.datasets.split(','):
        for model in args.models.split(','):
            notebook=next((ROOT/dataset).glob(f'*_{mapping[model]}_BASELINE_RUN_ALL.ipynb'))
            history=[];shell=SimpleNamespace(history_manager=SimpleNamespace(input_hist_raw=history))
            namespace={'torch':__import__('torch'),'get_ipython':lambda:shell,
                       'display':lambda x:print(x.to_string(index=False) if hasattr(x,'to_string') else x)}
            log=output/f'{dataset}_{model}_execution.txt'
            with log.open('w',encoding='utf-8') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                print('Actual delivered notebook:',notebook)
                print('Tabular: all released roots, one seed. Neural: balanced8roots/role, fanout2x2,2epochs, one seed.')
                try:
                    cells=json.loads(notebook.read_text(encoding='utf-8'))['cells']
                    for cell in cells:
                        if cell['cell_type']!='code':continue
                        source=''.join(cell['source']);tags=cell.get('metadata',{}).get('tags',[])
                        if 'environment' in tags:continue
                        history.append(source);exec(compile(source,str(notebook),'exec'),namespace)
                        if 'configuration' in tags:
                            work=output/dataset/model
                            namespace.update(RUN_MODE='smoke',DATA_RAW=Path(args.raw),
                                DATA_PHISH=Path(args.processed_root)/dataset/'data.pt',
                                OUTPUT_ROOT=output,WORK_ROOT=work,DATA_OUTPUT=work/'data',JOB_DIR=work/'results/smoke',DEVICE='cpu')
                        if 'prepare-data' in tags and model in ('gat','sage','metaifd'):
                            membership={}
                            for role,frame in namespace['membership'].items():
                                membership[role]=frame.groupby('label',group_keys=False).head(4).copy()
                            namespace['membership']=membership
                            namespace['data_manifest']['validation_only_reduced_roots']={r:len(p) for r,p in membership.items()}
                            print('Explicit validation-only reduction:',namespace['data_manifest']['validation_only_reduced_roots'])
                    summary=dict(dataset=dataset,model=model,status='PASS',delivered_notebook=str(notebook),
                        scientific_functions_mocked=False,run_mode='smoke',fits=len(namespace['model_lock']['fits']),
                        actual_root_counts={r:len(p) for r,p in namespace['membership'].items()},
                        max_neural_epochs=namespace['study']['protocol']['max_epochs'],
                        completed_notebook_execution_cells=True,full_five_seed_benchmark=False)
                    (output/f'{dataset}_{model}_validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
                except Exception:
                    traceback.print_exc();raise
            print(dataset,model,'PASS',flush=True)
            namespace.clear();gc.collect()

if __name__=='__main__':main()
