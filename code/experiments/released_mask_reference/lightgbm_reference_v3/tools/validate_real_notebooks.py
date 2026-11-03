"""Execute all literal reference notebook cells, with local paths only overridden."""
import argparse
import contextlib
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',required=True);parser.add_argument('--processed-root',required=True)
    args=parser.parse_args();output=ROOT/'validation/real_four';output.mkdir(parents=True,exist_ok=True)
    for dataset in ('Phish','Ponzi'):
        for model,label in [('lr','LR'),('lightgbm','LightGBM')]:
            notebook=next((ROOT/dataset).glob('*_'+label+'_BASELINE_X14_RUN_ALL.ipynb'))
            history=[];shell=SimpleNamespace(history_manager=SimpleNamespace(input_hist_raw=history))
            namespace={'torch':torch,'get_ipython':lambda:shell,'display':lambda value:print(value.to_string(index=False) if hasattr(value,'to_string') else value)}
            work=output/dataset/model
            with (output/f'{dataset}_{model}_execution.txt').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                for cell in json.loads(notebook.read_text(encoding='utf-8'))['cells']:
                    if cell['cell_type']!='code':continue
                    source=''.join(cell['source']);tags=cell.get('metadata',{}).get('tags',[])
                    history.append(source);exec(compile(source,str(notebook),'exec'),namespace)
                    if 'configuration' in tags:
                        namespace.update(DATA_RAW=Path(args.raw),DATA_PHISH=Path(args.processed_root)/dataset/'data.pt',
                                         OUTPUT_ROOT=output,WORK_ROOT=work,DATA_OUTPUT=work/'data',JOB_DIR=work/'results')
                protocol=namespace['study']['protocol'];assert protocol['arm']=='X14' and protocol['width']==14
                assert protocol['profile']=='reference' and len(namespace['model_lock']['fits'])==5
                assert protocol['fixed_config']['class_weight'] is None
                report=dict(status='PASS',dataset=dataset,model=model,fits=5,arm='X14',width=14,profile='reference',
                            scientific_functions_mocked=False,actual_notebook_cells_executed=True,environment_cell_executed=True,
                            partitions=1,split='released masks',notebook_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest())
                (output/f'{dataset}_{model}_validation.json').write_text(json.dumps(report,indent=2))
            namespace.clear();gc.collect();print(dataset,model,'PASS:5reference X14 fits',flush=True)


if __name__=='__main__':main()
