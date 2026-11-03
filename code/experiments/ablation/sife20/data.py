"""Read two input datasets and compute fresh features; no fitted model inputs."""
from contextlib import contextmanager
import gc
import hashlib
import io
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd
import torch

from .features import extract_sife44,PURE36_INDICES,FEATURE_VERSION
from .utilities import sha256,json_hash,write_json,read_json,require


class HashedStream(io.RawIOBase):
    def __init__(self,stream):
        self.stream=stream;self.digest=hashlib.sha256()
    def readable(self):return True
    def readinto(self,buffer):
        data=self.stream.read(len(buffer));buffer[:len(data)]=data
        self.digest.update(data);return len(data)
    def close(self):
        self.stream.close();super().close()


def resolve_input(root,dataset,kind):
    """Accept a ZIP or extracted directory, including nested Kaggle dataset mounts."""
    root=Path(root).expanduser().resolve()
    if not root.exists():raise ValueError(f'Missing {kind} input: {root}')
    if root.is_file():return root
    if kind=='raw':
        archives=sorted(root.rglob('OneDrive_2026-09-18.zip'))
        if len(archives)==1:return archives[0]
        marker='EOA_labeled.csv' if dataset=='Phish' else 'CA_labeled.csv'
        folders=[p.parent.parent for p in root.rglob(marker) if p.parent.name==dataset+'_unbalance']
        if len(folders)==1:return folders[0]
    else:
        matches=[p for p in root.rglob('data.pt') if dataset.lower() in [s.lower() for s in p.parts[:-1]]]
        if len(matches)==1:return matches[0]
        archives=sorted(root.rglob('HEIG.zip'))
        if len(archives)==1:return archives[0]
    raise ValueError(f'Need one {dataset} {kind} source; set its exact path in configuration')


@contextmanager
def open_raw_member(source,member):
    """Open a canonical member from ZIP or the extracted Meta-IFD data folder."""
    source=Path(source)
    if source.is_file():
        with zipfile.ZipFile(source) as archive:
            with archive.open(member) as stream:yield stream
    else:
        prefix='Meta-IFD data/'
        if not member.startswith(prefix):raise ValueError('Invalid raw member')
        path=(source/member[len(prefix):]).resolve()
        if not path.is_relative_to(source.resolve()):raise ValueError('Unsafe raw member')
        with path.open('rb') as stream:yield stream


def obtain_graph(source,output,dataset,expected_hash):
    """Verify known graph bytes before PyG deserialization; use direct files read-only."""
    source=Path(source)
    if source.suffix.lower()=='.pt':path=source
    else:
        path=Path(output)/'graph/data.pt'
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_suffix('.pt.partial')
            with zipfile.ZipFile(source) as archive:
                candidates=[n for n in archive.namelist() if n.endswith(f'{dataset}/processed/data.pt')]
                if len(candidates)!=1:raise ValueError('Need one target graph inside HEIG.zip')
                with archive.open(candidates[0]) as stream,temporary.open('wb') as target:
                    shutil.copyfileobj(stream,target,8*1024*1024)
            if sha256(temporary)!=expected_hash:raise ValueError('Wrong extracted graph')
            os.replace(temporary,path)
    if sha256(path)!=expected_hash:raise ValueError('Processed graph is not the locked dataset release')
    return path


def verified_raw_cohort(source,dataset,reference):
    """Verify the raw labelled cohort table against its frozen byte hash."""
    name='EOA_labeled.csv' if dataset=='Phish' else 'CA_labeled.csv'
    member=f'Meta-IFD data/{dataset}_unbalance/{name}'
    expected=reference['raw_members'][member]
    with open_raw_member(source,member) as raw:
        content=raw.read()
    if len(content)!=expected['bytes'] or hashlib.sha256(content).hexdigest()!=expected['sha256']:
        raise ValueError('Raw labelled table changed')
    frame=pd.read_csv(io.BytesIO(content))
    column='name' if dataset=='Phish' else 'Address'
    addresses=frame[column].astype(str).str.lower().to_numpy()
    mapped=np.asarray(reference['addresses'])
    if not set(mapped)<=set(addresses):raise ValueError('Frozen address mapping is absent from raw labelled table')
    if dataset=='Phish':
        positive=set(addresses[:1207]);negative=set(addresses[1207:])
        for address,label in zip(mapped,reference['labels']):
            if address not in (positive if label else negative):raise ValueError('Raw/source cohort label-group mismatch')
    return expected['sha256']


def restore_float64_attributes(raw_source,output,reference):
    """Restore all six raw Ponzi arrays, checking raw bytes/order and exact NPY hashes."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report={}
    for key,entry in reference['attribute_provenance']['relations'].items():
        target=output/entry['file']
        if target.exists():
            if sha256(target)!=entry['sha256']:raise ValueError('Cached Ponzi sidecar changed')
        else:
            src,rel,dst=key.split('__')
            member=f'Meta-IFD data/Ponzi_unbalance/edge/{rel}_{src.lower()}_{dst.lower()}.csv'
            expected=reference['raw_members'][member]
            temporary=target.with_suffix('.npy.partial')
            values=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float64,shape=(entry['rows'],2))
            cursor=0
            print('Restore raw float64:',key,flush=True)
            with open_raw_member(raw_source,member) as raw:
                reader=HashedStream(raw)
                with io.BufferedReader(reader,buffer_size=8*1024*1024) as stream:
                    for frame in pd.read_csv(stream,usecols=['count','sum'],chunksize=300000):
                        stop=cursor+len(frame)
                        if stop>len(values):raise ValueError('Too many raw rows')
                        chunk=frame[['count','sum']].to_numpy(np.float64)
                        if not np.isfinite(chunk).all() or (chunk<0).any():raise ValueError('Invalid raw edge weights')
                        values[cursor:stop]=chunk;cursor=stop
            if cursor!=len(values) or reader.digest.hexdigest()!=expected['sha256']:raise ValueError('Raw CSV checksum/row mismatch')
            values.flush();values._mmap.close();del values
            if sha256(temporary)!=entry['sha256']:raise ValueError('Restored array differs from raw-to-graph audit')
            os.replace(temporary,target)
        report[key]=entry['sha256']
    return report


def prepare_data(raw_source,processed_source,output,dataset,reference,source_signature):
    """Verify two data inputs, restore Ponzi weights, compute roots and remove duplicate addresses."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=obtain_graph(processed_source,output,dataset,reference['graph_sha256'])
    raw_hash=verified_raw_cohort(raw_source,dataset,reference)
    payload=torch.load(str(path),map_location='cpu',mmap=True,weights_only=False)
    graph=payload[0] if isinstance(payload,tuple) else payload
    require(set(graph.node_types)=={'EOA','CA'},'Expected released EOA/CA graph')
    require(all(graph[nt].x.shape[1]==14 for nt in graph.node_types),'Expected14source columns')
    target='EOA' if dataset=='Phish' else 'CA'
    ids=np.asarray(reference['ids'],dtype=np.int64);labels=np.asarray(reference['labels'],dtype=np.int64)
    addresses=np.asarray(reference['addresses'])
    require(np.array_equal(graph[target].y[torch.from_numpy(ids)].numpy(),labels),'Graph labels changed')
    sidecars=restore_float64_attributes(raw_source,output/'float64_attributes',reference) if dataset=='Ponzi' else {}
    edges={}
    for et in graph.edge_types:
        if dataset=='Ponzi':
            entry=reference['attribute_provenance']['relations']['__'.join(et)]
            attributes=np.load(output/'float64_attributes'/entry['file'],mmap_mode='r',allow_pickle=False)
        else:attributes=graph[et].edge_attr.numpy()
        edges[et]=dict(edge_index=graph[et].edge_index.numpy(),edge_attr=attributes)
    inputs=dict(dataset=dataset,graph_sha256=reference['graph_sha256'],raw_label_sha256=raw_hash,
        reference_hash=json_hash(reference),source_signature=source_signature,feature_version=FEATURE_VERSION,
        sidecar_hashes=sidecars)
    feature_path=output/'roots.npz';manifest_path=output/'data_manifest.json'
    if manifest_path.exists():
        manifest=read_json(manifest_path)
        require(manifest['inputs']==inputs and sha256(feature_path)==manifest['roots_sha256'],
                'Data/code cache changed; use a new work folder')
        with np.load(feature_path,allow_pickle=False) as stored:roots={key:stored[key].copy() for key in stored.files}
    else:
        _,inverse,counts=np.unique(addresses,return_inverse=True,return_counts=True)
        keep=counts[inverse]==1
        require(int(keep.sum())=={'Phish':2761,'Ponzi':1342}[dataset],'Unexpected strict cohort size')
        kept_ids=ids[keep]
        print('Compute exact root SIFE44:',dataset,len(kept_ids),'roots',flush=True)
        result=extract_sife44({nt:graph[nt].num_nodes for nt in graph.node_types},edges,target,kept_ids)
        roots=dict(ids=kept_ids,labels=labels[keep],addresses=addresses[keep],
                   source14=graph[target].x[torch.from_numpy(kept_ids)].numpy(),
                   sife44=result['sife44'],activity=result['activity'])
        np.savez_compressed(feature_path,**roots)
        manifest=dict(inputs=inputs,roots_sha256=sha256(feature_path),cohort_size=len(kept_ids),
            positives=int(labels[keep].sum()),excluded_ids=ids[~keep].tolist(),
            cohort_policy='exclude every supervised row belonging to any duplicated labelled address; graph topology unchanged',
            mapping_scope='audited labelled-root mapping bound to graph/raw byte hashes; existing raw-to-graph alignment audit',
            setting='static transductive; no verified timestamps or prediction cutoff',
            feature_scope='full incident typed edge records; same numeric definition for all models')
        write_json(manifest_path,manifest)
    require(len(set(roots['addresses']))==len(roots['ids']),'Strict root cohort contains duplicate addresses')
    return dict(graph=graph,edges=edges,roots=roots,manifest=manifest,
                node_counts={nt:graph[nt].num_nodes for nt in graph.node_types},data_output=output)

