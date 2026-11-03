"""Shared sampled topology and exact neighbor features; preprocessing fits TRAIN only."""
from pathlib import Path
import numpy as np
import torch
from .features import extract_sife44
from .utilities import require,sha256,read_json,write_json,seal_files,verify_files
from .protocol import TABULAR
from .sampling import save_pyg_role_batches,load_role_batches
from .views import materialize_decoder,save_meta_views


def prepare_role(study,split,role):
    """Materialize a role once; TEST is gated by all-partition fit locks."""
    require(role in ('train','validation','test'),'Unknown role')
    if role=='test':
        path=study['output']/'study_lock.json'
        require(path.exists(),'TEST cache is forbidden before the whole-study fit lock')
        lock=read_json(path)
        require(lock['identity']==study['identity'] and
                lock['split_lock_hashes'].get(str(split['seed']))==sha256(split['folder']/'lock.json'),
                'Test gating lock mismatch')
    root=split['folder']/'cache'/role;manifest_path=root/'manifest.json'
    if manifest_path.exists():
        manifest=read_json(manifest_path);require(manifest['identity']==split['identity'],'Cache identity changed')
        verify_files(root,manifest['files'])
        return [dict(path=str(root/'roots.npz'))] if study['protocol']['model'] in TABULAR else load_role_batches(root/'batches')
    root.mkdir(parents=True,exist_ok=True);p=study['protocol'];data=study['data']
    if p['model'] in TABULAR:
        roots=data['roots'];pos=split['parts'][role]
        np.savez_compressed(root/'roots.npz',ids=roots['ids'][pos],labels=roots['labels'][pos],
                            source14=roots['source14'][pos],sife44=roots['sife44'][pos])
        records=[dict(path=str(root/'roots.npz'))]
    else:
        seed=dict(train=1201,validation=1202,test=2301)[role]+10000*(split['seed']-2000)
        print('Cache native graph:',p['dataset'],p['model'],split['seed'],role,flush=True)
        records=save_pyg_role_batches(data['graph'],split['membership'][role],p['target'],role,
                                     root/'batches',seed,p['fanouts'],p['batch_size'])
        feature_dir=root/'features';feature_dir.mkdir(exist_ok=True)
        by_type={nt:[] for nt in ('EOA','CA')}
        for row in records:
            batch=torch.load(row['path'],map_location='cpu',weights_only=True)
            for nt in by_type:by_type[nt].append(batch['node_ids'][nt].numpy())
        for nt,parts in by_type.items():
            ids=np.unique(np.concatenate(parts))
            print('Exact incident features:',nt,len(ids),'nodes',flush=True)
            values=(extract_sife44(data['node_counts'],data['edges'],nt,ids)['sife44'] if len(ids) else np.empty((0,44)))
            np.save(feature_dir/(nt+'_ids.npy'),ids)
            np.save(feature_dir/(nt+'_sife44.npy'),values)
        if p['model']=='metaifd':
            decoder=materialize_decoder(study['output']/'reference',study['decoder_reference'])
            save_meta_views(data['graph'],records,root/'views',decoder,seed,p['config']['views'])
            del decoder
    files=[path for path in root.rglob('*') if path.is_file() and not path.name.endswith('.partial')]
    write_json(manifest_path,dict(identity=split['identity'],role=role,files=seal_files(root,files),
        feature_scope='full incident graph records for unique role nodes; labels never enter feature construction'))
    return records


def fit_training_statistics(study,split):
    """Two-pass float64 moments on unique typed TRAIN neighbors; source columns are independent of extras."""
    data=study['data'];root=split['folder']/'cache/train/features'
    total=np.zeros(58,dtype=np.float64);count=0;typed_counts={}
    for nt in ('EOA','CA'):
        ids=np.load(root/(nt+'_ids.npy'),mmap_mode='r',allow_pickle=False)
        extra=np.load(root/(nt+'_sife44.npy'),mmap_mode='r',allow_pickle=False)
        typed_counts[nt]=len(ids);count+=len(ids)
        for start in range(0,len(ids),100000):
            index=torch.tensor(np.array(ids[start:start+100000],copy=True),dtype=torch.long)
            source=data['graph'][nt].x[index].numpy().astype(np.float64)
            if study['protocol']['model']=='gat':source=np.sign(source)*np.log1p(np.abs(source))
            total[:14]+=source.sum(0);total[14:]+=extra[start:start+100000].sum(0)
    require(count>0,'No TRAIN nodes to fit preprocessing')
    mean=total/count;squared=np.zeros(58,dtype=np.float64)
    for nt in ('EOA','CA'):
        ids=np.load(root/(nt+'_ids.npy'),mmap_mode='r',allow_pickle=False)
        extra=np.load(root/(nt+'_sife44.npy'),mmap_mode='r',allow_pickle=False)
        for start in range(0,len(ids),100000):
            index=torch.tensor(np.array(ids[start:start+100000],copy=True),dtype=torch.long)
            source=data['graph'][nt].x[index].numpy().astype(np.float64)
            if study['protocol']['model']=='gat':source=np.sign(source)*np.log1p(np.abs(source))
            squared[:14]+=((source-mean[:14])**2).sum(0)
            squared[14:]+=((extra[start:start+100000]-mean[14:])**2).sum(0)
    scale=np.sqrt(squared/count);constant=scale<1e-12;scale[constant]=1.
    require(np.isfinite(mean).all() and np.isfinite(scale).all(),'Nonfinite TRAIN preprocessing')
    # One float32 representation is also used for Meta/SAGE extras, matching model input precision.
    statistics=dict(mean=mean.astype(np.float32).tolist(),scale=scale.astype(np.float32).tolist(),
        unique_typed_train_nodes=count,counts_by_type=typed_counts,labels_used=False,
        scope='unique typed nodes in TRAIN sampled cache; transductive unlabeled neighbors included',
        constant_columns=np.flatnonzero(constant).tolist(),population_sd=True,
        source_transform='signed_log1p' if study['protocol']['model']=='gat' else 'identity',
        extras_transform='identity before frozen TRAIN z-score; Pure36 selects only its36 columns')
    path=split['folder']/'input_statistics.json'
    if path.exists():require(read_json(path)==statistics,'TRAIN statistics changed')
    else:write_json(path,statistics)
    split['statistics']=statistics
    return statistics
