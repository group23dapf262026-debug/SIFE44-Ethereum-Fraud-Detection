"""Verify release bytes and keep the released masks without cohort cleaning."""
from contextlib import contextmanager
import hashlib
import io
import os
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd
import torch
from .utilities import require,sha256,json_hash,read_json,write_json


def resolve_input(root,dataset,kind):
    """Locate one raw ZIP/folder or one processed graph/HEIG.zip."""
    root=Path(root).expanduser().resolve()
    require(root.exists(),f'Missing {kind} input: {root}')
    if root.is_file():return root
    if kind=='raw':
        matches=list(root.rglob('OneDrive_2026-09-18.zip'))
        if len(matches)==1:return matches[0]
        marker='EOA_labeled.csv' if dataset=='Phish' else 'CA_labeled.csv'
        matches=[p.parent.parent for p in root.rglob(marker) if p.parent.name==dataset+'_unbalance']
    else:
        matches=[p for p in root.rglob('data.pt') if dataset.lower() in [x.lower() for x in p.parts[:-1]]]
        if len(matches)==1:return matches[0]
        matches=list(root.rglob('HEIG.zip'))
    require(len(matches)==1,f'Set the exact {dataset} {kind} path in Configuration; found {matches}')
    return matches[0]


@contextmanager
def open_raw_member(source,member):
    """Read the same labelled table from a ZIP or extracted raw folder."""
    source=Path(source)
    if source.is_file():
        with zipfile.ZipFile(source) as archive:
            with archive.open(member) as stream:yield stream
    else:
        require(member.startswith('Meta-IFD data/'),'Invalid raw member')
        path=(source/member[len('Meta-IFD data/'):]).resolve()
        require(path.is_relative_to(source.resolve()),'Unsafe raw path')
        with path.open('rb') as stream:yield stream


def obtain_graph(source,output,dataset,expected_hash):
    """Hash-check the known graph before deserialization; direct files remain read-only."""
    source=Path(source)
    if source.suffix.lower()=='.pt':path=source
    else:
        path=Path(output)/'graph/data.pt'
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_suffix('.partial')
            with zipfile.ZipFile(source) as archive:
                names=[n for n in archive.namelist() if n.endswith(f'{dataset}/processed/data.pt')]
                require(len(names)==1,'Expected one target processed graph in ZIP')
                with archive.open(names[0]) as stream,temporary.open('wb') as dest:
                    shutil.copyfileobj(stream,dest,8*1024*1024)
            require(sha256(temporary)==expected_hash,'Wrong extracted graph bytes')
            os.replace(temporary,path)
    require(sha256(path)==expected_hash,'Wrong graph release; do not deserialize it')
    return path


def released_membership(graph,reference,dataset):
    """Use graph train_mask/val_mask/test_mask verbatim and disclose address overlap."""
    target='EOA' if dataset=='Phish' else 'CA'
    ids=np.asarray(reference['ids'],dtype=np.int64)
    labels=np.asarray(reference['labels'],dtype=np.int64)
    require(np.array_equal(graph[target].y[torch.from_numpy(ids)].numpy(),labels),'Release label alignment changed')
    cohort=pd.DataFrame(dict(node_id=ids,label=labels,address=reference['addresses'])).set_index('node_id')
    parts={}
    for role,mask in [('train','train_mask'),('validation','val_mask'),('test','test_mask')]:
        selected=torch.where(graph[target][mask])[0].numpy()
        require(set(selected)<=set(ids),'Mask includes unknown labelled IDs')
        parts[role]=cohort.loc[selected].reset_index()
    combined=np.concatenate([p.node_id.to_numpy() for p in parts.values()])
    require(len(combined)==len(set(combined))==len(ids) and set(combined)==set(ids),'Released masks overlap or miss roots')
    address_sets={r:set(p.address) for r,p in parts.items()}
    overlaps={a+'_'+b:sorted(address_sets[a]&address_sets[b]) for a,b in
              [('train','validation'),('train','test'),('validation','test')]}
    expected={'Phish':{'train':1657,'validation':553,'test':553},
              'Ponzi':{'train':804,'validation':269,'test':269}}
    sizes={r:len(p) for r,p in parts.items()}
    require(sizes==expected[dataset],'Unexpected released split sizes')
    return parts,dict(dataset=dataset,target=target,rule='unchanged released masks; no resplitting',
        sizes=sizes,positives={r:int(p.label.sum()) for r,p in parts.items()},
        address_intersections=overlaps,unique_addresses=int(cohort.address.nunique()),
        removed_ids=[],exact_paper_table_run_recovered=False)


def prepare_release(raw_source,processed_source,output,dataset,reference):
    """Verify two inputs and source14; no incident features, raw edge restoration or fitting."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=obtain_graph(processed_source,output,dataset,reference['graph_sha256'])
    member=f'Meta-IFD data/{dataset}_unbalance/'+('EOA_labeled.csv' if dataset=='Phish' else 'CA_labeled.csv')
    with open_raw_member(raw_source,member) as stream:content=stream.read()
    expected=reference['raw_members'][member]
    require(len(content)==expected['bytes'] and hashlib.sha256(content).hexdigest()==expected['sha256'],
            'Raw labelled table checksum mismatch')
    frame=pd.read_csv(io.BytesIO(content))
    addresses=frame['name' if dataset=='Phish' else 'Address'].astype(str).str.lower().to_numpy()
    require(set(reference['addresses'])<=set(addresses),'Mapped roots missing in raw table')
    if dataset=='Phish':
        groups={1:set(addresses[:1207]),0:set(addresses[1207:])}
        require(all(a in groups[int(y)] for a,y in zip(reference['addresses'],reference['labels'])),
                'Phish label groups differ from audited mapping')
    payload=torch.load(str(path),map_location='cpu',mmap=True,weights_only=False)
    graph=payload[0] if isinstance(payload,tuple) else payload
    require(set(graph.node_types)=={'EOA','CA'},'Unexpected node types')
    require(all(graph[nt].x.shape[1]==14 for nt in graph.node_types),'Source width must be14')
    membership,report=released_membership(graph,reference,dataset)
    manifest=dict(dataset=dataset,graph_sha256=reference['graph_sha256'],raw_label_sha256=expected['sha256'],
                  address_mapping_reference_sha256=json_hash(reference),source14='unchanged released graph values',
                  mapping_scope='audited labelled-root mapping bound to known graph and raw table bytes; not rebuilding all raw nodes',
                  split_report=report)
    write_json(output/'data_manifest.json',manifest)
    for role,part in membership.items():part.to_csv(output/(role+'_membership.csv'),index=False)
    return graph,membership,manifest
