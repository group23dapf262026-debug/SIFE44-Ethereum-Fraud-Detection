"""Copy audited local reference definitions once, with explicit provenance."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent
OLD=PARENT/'Phish_FIXED_v3'/'sife44'

def take(module,names):
    source=(OLD/(module+'.py')).read_text(encoding='utf-8')
    lines=source.splitlines();out=[]
    for node in ast.parse(source).body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            if isinstance(node,ast.ImportFrom) and node.level:continue
            out.append(ast.get_source_segment(source,node))
        elif isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
            start=min([node.lineno]+[d.lineno for d in node.decorator_list])
            out.append('\n'.join(lines[start-1:node.end_lineno]))
    return '\n\n'.join(out)+'\n'

def main():
    (ROOT/'baseline').mkdir(parents=True,exist_ok=True)
    (ROOT/'assets').mkdir(exist_ok=True)
    (ROOT/'baseline/__init__.py').write_text('',encoding='utf-8')
    utility_names=['require','sha256','json_hash','write_json','read_json','seed_everything',
                   'runtime_information','seal_files','verify_files','atomic_torch_save']
    (ROOT/'baseline/utilities.py').write_text(take('utilities',utility_names),encoding='utf-8')
    (ROOT/'baseline/metrics.py').write_text('from .utilities import require\n'+take('metrics',['classification_metrics']),encoding='utf-8')
    # Retain only actual PyG sampling, not the older optional NumPy sampler.
    sampling=take('sampling',['save_pyg_role_batches','load_role_batches'])
    sampling='from .utilities import require, sha256, read_json, write_json, atomic_torch_save\n'+sampling
    (ROOT/'baseline/sampling.py').write_text(sampling,encoding='utf-8')
    for name in ['attention_conv','models_meta']:
        source=(OLD/(name+'.py')).read_text(encoding='utf-8')
        if name=='models_meta':
            source=source.replace('input_width=14','input_width=14')
            source=source.replace('def forward(self, x_dict, edge_index, sife_dict=None):','def forward(self, x_dict, edge_index):')
            start=source.index('                if sife_dict is None:')
            end=source.index("                x_dict_[node_type]",start)
            source=source[:start]+'                encoded = layer(x[k])\n'+source[end:]
        (ROOT/'baseline'/(name+'.py')).write_text(source,encoding='utf-8')
    # Plain release mappings are data, not precomputed features or executable payloads.
    for ds in ['Phish','Ponzi']:
        source=PARENT/'SIFE_SOFTT_2026/assets'/f'release_reference_{ds}.json'
        ref=json.loads(source.read_text(encoding='utf-8'))
        minimal={k:ref[k] for k in ['graph_sha256','ids','labels','addresses','raw_members']}
        (ROOT/'assets'/source.name).write_text(json.dumps(minimal,indent=2),encoding='utf-8')
    decoder=PARENT/'Phish_FIXED_v3/assets/decoder_tensor_values.json'
    (ROOT/'assets/decoder_tensor_values.json').write_bytes(decoder.read_bytes())
    provenance={str(p.relative_to(PARENT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [OLD/'attention_conv.py',OLD/'models_meta.py',OLD/'sampling.py',decoder]}
    (ROOT/'REFERENCE_SOURCE_HASHES.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')

if __name__=='__main__':main()
