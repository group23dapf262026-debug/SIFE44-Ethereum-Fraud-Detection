"""Run a readable follow-up notebook with explicit portable data locations."""
from pathlib import Path
import argparse
import json
import os
import re


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook',type=Path)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--processed',type=Path,required=True)
    parser.add_argument('--work-root',type=Path,required=True)
    parser.add_argument('--mode',choices=['smoke','full'],default='smoke')
    parser.add_argument('--device',default=None)
    parser.add_argument('--max-hours',type=float,default=8)
    parser.add_argument('--no-install',action='store_true')
    parser.add_argument('--full-cache-export',action='store_true')
    args=parser.parse_args()
    for path in (args.notebook,args.raw,args.processed):
        if not path.exists():parser.error(f'Input does not exist: {path}')
    if args.max_hours<=0:parser.error('--max-hours must be positive')
    os.environ['SIFE20_RAW_INPUT']=str(args.raw.resolve())
    os.environ['SIFE20_PROCESSED_INPUT']=str(args.processed.resolve())
    # Configuration appends experiment/dataset/model, preventing cross-experiment reuse.
    os.environ['SIFE20_WORK_ROOT']=str(args.work_root.resolve()/args.mode)
    from IPython.core.interactiveshell import InteractiveShell
    shell=InteractiveShell.instance()
    shell.user_ns.update(__name__='__main__',__file__=str(args.notebook.resolve()))
    book=json.loads(args.notebook.read_text(encoding='utf-8'))
    for i,cell in enumerate(book['cells']):
        if cell['cell_type']!='code':continue
        source=''.join(cell['source'])
        if args.no_install:source=source.replace('INSTALL_MISSING_PACKAGES=True','INSTALL_MISSING_PACKAGES=False')
        configuration=bool(re.search(r'^RUN_MODE=',source,re.M))
        if configuration:
            source=re.sub(r"^RUN_MODE='[^']*'",f'RUN_MODE={args.mode!r}',source,flags=re.M)
            source=re.sub(r'^MAX_SESSION_HOURS=8\b',f'MAX_SESSION_HOURS={args.max_hours!r}',source,flags=re.M)
            if args.full_cache_export:
                source=source.replace('INCLUDE_LARGE_CACHES_IN_EXPORT=False','INCLUDE_LARGE_CACHES_IN_EXPORT=True')
        print(f'Executing {args.notebook.name}: code cell {i}',flush=True)
        result=shell.run_cell(source,store_history=True)
        result.raise_error()
        if configuration and args.device:shell.user_ns['DEVICE']=args.device
    state='Notebook paused with saved progress.' if shell.user_ns.get('FITTING_COMPLETE') is False else 'Notebook completed.'
    print(state,'Work base:',args.work_root.resolve(),flush=True)


if __name__=='__main__':main()
