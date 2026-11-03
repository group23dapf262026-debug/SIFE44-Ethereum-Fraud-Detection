"""Execute a standalone experiment notebook on local, Colab, or server inputs."""
from pathlib import Path
import argparse
import json
import os
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', type=Path, help='Path to one of the standalone .ipynb files')
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--processed', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--device', help='For example cpu or cuda:0; omitted uses notebook auto-detection')
    parser.add_argument('--no-install', action='store_true', help='Disable the notebook package installer')
    args = parser.parse_args()
    path = args.notebook.resolve()
    for item in [path, args.raw, args.processed]:
        if not item.exists(): parser.error(f'Input does not exist: {item}')
    book = json.loads(path.read_text(encoding='utf-8'))
    has_run_mode = any(re.search(r'^RUN_MODE\s*=', ''.join(c.get('source', [])), re.M)
                       for c in book['cells'] if c['cell_type'] == 'code')
    if args.mode == 'smoke' and not has_run_mode:
        parser.error('This released-mask reference notebook has no reduced smoke mode; use --mode full explicitly.')
    # These variables change input locations only. Keep full/smoke roots separate.
    stage = 'ablation' if '/ablation/' in path.as_posix() else ('reference' if 'released_mask_reference' in path.as_posix() else 'main')
    root = (args.work_root.resolve() / stage / args.mode)
    root.mkdir(parents=True, exist_ok=True)
    for key in ['SIFE20_RAW_INPUT', 'BASELINE_RAW_INPUT', 'RAW_INPUT']:
        os.environ[key] = str(args.raw.resolve())
    for key in ['SIFE20_PROCESSED_INPUT', 'BASELINE_PROCESSED_INPUT', 'PROCESSED_INPUT']:
        os.environ[key] = str(args.processed.resolve())
    for key in ['SIFE20_WORK_ROOT', 'SIFE20_ABLATION_WORK_ROOT', 'BASELINE_WORK_ROOT']:
        os.environ[key] = str(root)
    os.environ['SIFE_RAW_INPUT'] = str(args.raw.resolve())
    os.environ['SIFE_PROCESSED_INPUT'] = str(args.processed.resolve())
    # Frozen source identities are collected from IPython's real input history.
    # Plain exec would leave get_ipython() unset and break the scientific lock.
    from IPython.core.interactiveshell import InteractiveShell
    shell = InteractiveShell.instance()
    context = shell.user_ns
    context.update({'__name__': '__main__', '__file__': str(path)})
    # The baseline notebooks also accept explicit configuration variables; set
    # paths after their configuration cell without editing the stored notebook.
    for index, cell in enumerate(book['cells']):
        if cell['cell_type'] != 'code': continue
        source = ''.join(cell.get('source', []))
        if args.no_install:
            source = source.replace('INSTALL_MISSING_PACKAGES=True', 'INSTALL_MISSING_PACKAGES=False')
        if re.search(r'^RUN_MODE\s*=', source, re.M):
            source = re.sub(r"^RUN_MODE\s*=\s*['\"][^'\"]+['\"]", f'RUN_MODE={args.mode!r}', source, flags=re.M)
        print(f'Executing {path.name}: code cell {index}', flush=True)
        result = shell.run_cell(source, store_history=True)
        result.raise_error()
        if re.search(r'^DATA_RAW\s*=', source, re.M):
            context['DATA_RAW'] = args.raw.resolve(); context['DATA_PHISH'] = args.processed.resolve()
            if args.device: context['DEVICE'] = args.device
    print('Notebook completed. Work root:', root, flush=True)

if __name__ == '__main__': main()
