"""Verify the compact research artifact without Torch, GPU, or external services."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
VIETNAMESE = re.compile(r'[\u0102-\u0103\u0110-\u0111\u0128-\u0129\u0168-\u0169\u01a0-\u01b0\u1ea0-\u1ef9]')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def main(repo, anonymous_view=False):
    manifest = repo / 'MANIFEST.sha256'
    assert manifest.exists(), 'MANIFEST.sha256 is missing'
    included = []
    presentation_files = {'README.md', 'CITATION.cff', 'LICENSE', 'LICENSE.md',
                          'THIRD_PARTY_NOTICES.md', 'CODE_LICENSE_SCOPE.json'}
    presentation_files.update(p.relative_to(repo).as_posix() for p in (repo/'docs').glob('*.md'))
    skipped_presentation = []
    for line in manifest.read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ', 1)
        path = (repo / name).resolve()
        assert path.is_relative_to(repo.resolve()), 'Unsafe manifest path'
        assert path.is_file(), f'Missing file: {name}'
        if anonymous_view and name in presentation_files:
            skipped_presentation.append(name)
        else:
            assert sha(path) == expected, f'Checksum mismatch: {name}'
        included.append(name)
    actual = {p.relative_to(repo).as_posix() for p in repo.rglob('*') if p.is_file()
              and not any(x in p.relative_to(repo).parts for x in ['generated', '__pycache__', '.git'])}
    assert actual == set(included) | {'MANIFEST.sha256'}, 'Unlisted or obsolete files remain in the artifact'
    books = json.loads((repo / 'provenance/notebook_code_identity.json').read_text(encoding='utf-8'))
    assert len(books) == 30
    notebook_matrix = []
    checked_signatures = 0
    for row in books:
        path = repo / row['path']; book = json.loads(path.read_text(encoding='utf-8'))
        cells = [''.join(c.get('source', [])) for c in book['cells'] if c['cell_type'] == 'code']
        assert hashlib.sha256(json.dumps(cells, ensure_ascii=False).encode('utf-8')).hexdigest() == row['code_cells_sha256']
        for source in cells: ast.parse(source)
        assert all(c.get('execution_count') is None and not c.get('outputs') for c in book['cells'] if c['cell_type'] == 'code')
        all_text = '\n'.join(''.join(c.get('source', [])) for c in book['cells'])
        assert not VIETNAMESE.search(all_text), f'Non-English Vietnamese text in {path.name}'
        assert 'â€' not in all_text and 'Ã—' not in all_text, 'Mojibake remains'
        dataset = re.search(r"^DATASET=['\"]([^'\"]+)", '\n'.join(cells), re.M).group(1)
        model = re.search(r"^MODEL=['\"]([^'\"]+)", '\n'.join(cells), re.M).group(1)
        assert model in ['metaifd', 'gat', 'sage', 'lightgbm', 'rf']
        stage = 'ablation' if '/ablation/' in row['path'] else ('baseline' if 'released_mask_reference' in row['path'] else 'main')
        original = json.loads((repo / 'results' / stage / 'runs' / f'{dataset.lower()}_{model}' / 'protocol.json').read_text(encoding='utf-8'))
        expected = original.get('source_signature') or original.get('source')
        signatures = {}
        for cell in cells:
            lines = cell.splitlines()
            if lines and re.match(r'# (?:SIFE20|BASELINE)_SOURCE_CELL: ', lines[0]):
                # The LightGBM reference signs literal cells; the native notebook
                # implementation signs IPython history with trailing whitespace stripped.
                signed = cell if 'lightgbm_reference_v3' in row['path'] else cell.rstrip()
                signatures[lines[0].split(': ', 1)[1]] = hashlib.sha256(signed.encode()).hexdigest()
        if isinstance(expected, dict) and set(expected).issubset(signatures):
            assert all(signatures[key] == value for key, value in expected.items()), f'Frozen source mismatch: {path.name}'
            checked_signatures += 1
        elif stage in ['main', 'ablation']:
            raise AssertionError(f'Could not verify frozen source identity: {path.name}')
        notebook_matrix.append(dict(stage=stage, dataset=dataset, model=model, path=row['path'], code_cells=len(cells)))
    assert checked_signatures == 30
    py_count = 0
    for path in (repo / 'code').rglob('*.py'):
        text = path.read_text(encoding='utf-8-sig')
        ast.parse(text, filename=str(path)); py_count += 1
        assert not VIETNAMESE.search(text), f'Non-English Vietnamese text in {path}'
        assert not re.search(r'[CD]:\\', text), f'Machine-specific drive path in {path}'
    broken = []
    for path in repo.rglob('*.md'):
        if 'generated' in path.parts: continue
        text = path.read_text(encoding='utf-8')
        assert not VIETNAMESE.search(text), f'Non-English Vietnamese text in {path}'
        for link in re.findall(r'\[[^]]*\]\(([^)]+)\)', text):
            if re.match(r'\w+://|mailto:|#', link): continue
            target = (path.parent / link.split('#')[0]).resolve()
            if not target.exists(): broken.append((str(path.relative_to(repo)), link))
    assert not broken, broken
    # A scoped grant must not be mistaken for relicensing the upstream model.
    license_scope = json.loads((repo / 'CODE_LICENSE_SCOPE.json').read_text(encoding='utf-8'))
    assert license_scope['spdx_identifier'] == 'MIT'
    assert license_scope['upstream_permission_confirmed'] is False
    assert (repo / 'LICENSE').read_text(encoding='utf-8').startswith('MIT License')
    for item in license_scope['full_files']:
        if not (anonymous_view and item['path'] in presentation_files):
            assert sha(repo / item['path']) == item['sha256'], f'License scope file changed: {item["path"]}'
        assert not item['path'].endswith(('models_meta.py', 'attention_conv.py', 'views.py'))
    for item in license_scope['mixed_notebook_feature_cells']:
        book = json.loads((repo / item['notebook']).read_text(encoding='utf-8'))
        for entry in item['covered_cells']:
            text = ''.join(book['cells'][entry['index']]['source'])
            assert text.startswith('# SIFE20_SOURCE_CELL: features.')
            assert hashlib.sha256(text.encode('utf-8')).hexdigest() == entry['sha256']
    report = dict(status='PASS', manifest_files=len(included), notebooks=30,
                  frozen_notebook_signatures_verified=checked_signatures, python_files_parsed=py_count,
                  notebook_code_cells_unchanged=True, English_sources_checked=True,
                  broken_local_document_links=broken, scoped_license_checks=True,
                  upstream_relicensed=False, training_performed=False,
                  anonymous_view=anonymous_view, identity_presentation_hashes_skipped=skipped_presentation,
                  scientific_source_data_and_notebook_hashes_still_required=True)
    print(json.dumps(report, indent=2))
    return report

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=ROOT)
    p.add_argument('--anonymous-view', action='store_true',
                   help='Skip hashes only for explicitly listed identity-bearing presentation files changed by an anonymous mirror')
    args = p.parse_args(); main(args.repo.resolve(), args.anonymous_view)
