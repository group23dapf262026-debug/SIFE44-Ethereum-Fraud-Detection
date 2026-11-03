"""Fixed single-split X14 baseline: prepare -> fit -> lock -> evaluate."""
import gc
import hashlib
import itertools
from pathlib import Path
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits
from .utilities import require,sha256,json_hash,write_json,read_json,seed_everything,runtime_information,seal_files,verify_files,atomic_torch_save
from .sampling import save_pyg_role_batches,load_role_batches
from .models import build_model,forward_batch,materialize_decoder,save_meta_views
from .metrics import classification_metrics

MODELS=('metaifd','gat','sage','lightgbm','lr','rf')
TABULAR=('lightgbm','lr','rf')


def fixed_configuration(model_name):
    """One accepted-study configuration per model, chosen before this run; no grid."""
    configurations=dict(metaifd=dict(hidden=128,lr=.001,views=3,contrast_weight=.1),
        gat=dict(hidden=32,lr=.01),sage=dict(hidden=32,lr=.01),
        lightgbm=dict(n_estimators=300,lr=.01,num_leaves=31),
        lr=dict(C=10.,solver='liblinear',max_iter=5000),rf=dict(n_estimators=100,max_depth=10))
    require(model_name in configurations,'Only the six agreed models are supported')
    return configurations[model_name].copy()


def notebook_source_signature():
    """Bind the literal executed scientific cells to this experiment identity."""
    shell=get_ipython()
    require(shell is not None,'Run the notebook with a normal IPython/Jupyter kernel')
    latest={}
    for source in shell.history_manager.input_hist_raw:
        first=source.splitlines()[0] if source.splitlines() else ''
        prefix='# BASELINE_SOURCE_CELL: '
        if first.startswith(prefix):latest[first[len(prefix):]]=hashlib.sha256(source.encode()).hexdigest()
    require(set(DEFINITION_KEYS)<=set(latest),'Run all definition cells first')
    return {key:latest[key] for key in DEFINITION_KEYS}


def open_study(graph,membership,data_manifest,job_dir,dataset,model_name,device,source_signature,
               decoder_reference=None,run_mode='full',cpu_threads=2):
    """Freeze masks, fixed config, code, reference data, RNG and runtime before training."""
    require(model_name in MODELS and run_mode in ('full','smoke'),'Invalid model or run mode')
    require(set(membership)=={'train','validation','test'},'Missing split role')
    torch.set_num_threads(cpu_threads)
    neural=model_name not in TABULAR
    require(neural or str(device)=='cpu','Tabular models must use CPU')
    protocol=dict(version='released_x14_native_fixed_v2',dataset=dataset,model=model_name,arm='X14',width=14,
        cohort='released_reference; retain all released labelled rows, including known Phish conflict',
        split='unchanged graph masks; one partition; no resplitting',
        fixed_config=fixed_configuration(model_name),configuration_source=(
            'public utils.py defaults, also used by accepted study; not a recovered paper-table winner' if model_name=='metaifd' else
            'accepted-study fixed settings; not recovered paper-table winning config'),
        model_seeds=[12,13,14,15,16] if run_mode=='full' else [12],run_mode=run_mode,
        hyperparameter_search=False,decision='p1 > 0.5; exact tie -> class0; no threshold search',
        preprocessing=('raw source14; StandardScaler fitted only to TRAIN roots' if model_name=='lr' else
                       'raw source14; no external scaler' if model_name in ('rf','lightgbm') else
                       'source14 signed-log1p then frozen z-score from unique typed TRAIN sampled nodes' if model_name=='gat' else
                       'raw source14; learned Linear14->hidden then tanh; mean SAGE project=True' if model_name=='sage' else
                       'public decoder generated normalized interaction views + unchanged source14'),
        max_epochs=(1000 if run_mode=='full' else 2) if neural else None,
        patience=50 if neural else None,weight_decay=.0005 if neural else None,
        checkpoint_rule='latest validation Macro-F1 tie; stop stale >50' if neural else 'one fitted estimator at fixed budget',
        fanouts=[100,100] if run_mode=='full' else [2,2],batch_size=512 if run_mode=='full' else 8,
        sampler_seeds=dict(train=1201,validation=1202,test=2301),
        sampler='actual PyG NeighborLoader directional; per-role cached draws shared across model seeds' if neural else 'none',
        author_commit='439c6daa936f67f19486be76642e690a2153d0c5',
        meta_loss='public-code unsquared L2 TripletMarginLoss mean, margin .3; differs from squared/sum paper equation',
        test_policy='freeze all five seed estimators/checkpoints before test cache or test inference',
        setting='static transductive; topology and unlabeled held-out features may be sampled as neighbors',
        inference='descriptive seed variation on one released split; not independent-split uncertainty',
        exact_paper_table_reproduction_claimed=False)
    protocol['backbone_source']={
        'gat':'audited phish_readable_gat / phish_gat_signedlog_exact_env211_v2',
        'sage':'audited phish_readable_graphsage / phish_graphsage_mean_dataset_exact_env211_v3',
        'metaifd':'pinned public HMSL and decoder'}.get(model_name,'accepted-study tabular baseline')
    protocol['architecture_status']='group implementation; not an exact original-paper GAT/SAGE reconstruction' if model_name in ('gat','sage') else 'see deviations'
    identity=dict(protocol=protocol,data=data_manifest,runtime=runtime_information(device),source=source_signature,
        decoder_reference=json_hash(decoder_reference) if decoder_reference else None,
        membership={role:frame.to_dict('records') for role,frame in membership.items()})
    job=Path(job_dir);job.mkdir(parents=True,exist_ok=True)
    path=job/'protocol.json'
    if path.exists():require(read_json(path)==identity,'Code/config/data/runtime changed; use a new output folder')
    else:write_json(path,identity)
    write_json(job/'split_report.json',data_manifest['split_report'])
    for role,part in membership.items():part.to_csv(job/(role+'_membership.csv'),index=False)
    return dict(graph=graph,membership=membership,protocol=protocol,identity=json_hash(identity),
                job=job,device=torch.device(device),decoder_reference=decoder_reference)


def prepare_role(study,role):
    """Cache only requested roots/topology/views; test cache requires the model lock."""
    require(role in ('train','validation','test'),'Unknown role')
    if role=='test':verify_model_lock(study)
    job=study['job'];root=job/'cache'/role;manifest_path=root/'manifest.json'
    if manifest_path.exists():
        manifest=read_json(manifest_path)
        require(manifest['identity']==study['identity'],'Cache belongs to another experiment')
        verify_files(root,manifest['files'])
        return [dict(path=str(root/'roots.npz'))] if study['protocol']['model'] in TABULAR else load_role_batches(root/'batches')
    root.mkdir(parents=True,exist_ok=True)
    graph=study['graph'];protocol=study['protocol'];target='EOA' if protocol['dataset']=='Phish' else 'CA'
    membership=study['membership'][role]
    if protocol['model'] in TABULAR:
        ids=membership.node_id.to_numpy(dtype=np.int64,copy=True)
        values=graph[target].x[torch.from_numpy(ids)].numpy().astype(np.float64)
        require(values.shape==(len(ids),14) and np.isfinite(values).all(),'Invalid source14')
        np.savez_compressed(root/'roots.npz',ids=ids,labels=membership.label.to_numpy(dtype=np.int64),source14=values)
        records=[dict(path=str(root/'roots.npz'))]
    else:
        records=save_pyg_role_batches(graph,membership,target,role,root/'batches',
            protocol['sampler_seeds'][role],protocol['fanouts'],protocol['batch_size'])
        if protocol['model']=='metaifd':
            require(study['decoder_reference'] is not None,'Missing public decoder numeric reference')
            decoder=materialize_decoder(job/'reference',study['decoder_reference'])
            save_meta_views(graph,records,root/'views',decoder,protocol['sampler_seeds'][role],protocol['fixed_config']['views'])
            del decoder
    files=[p for p in root.rglob('*') if p.is_file() and not p.name.endswith('.partial')]
    write_json(manifest_path,dict(identity=study['identity'],files=seal_files(root,files)))
    return records


def make_estimator(model_name,config,seed,cpu_threads):
    """Match accepted-study tabular settings; scaler is part of the saved LR pipeline."""
    if model_name=='lr':
        return Pipeline([('scale',StandardScaler()),('estimator',LogisticRegression(
            C=config['C'],solver=config['solver'],max_iter=config['max_iter'],class_weight=None,random_state=seed))])
    if model_name=='rf':
        return RandomForestClassifier(n_estimators=config['n_estimators'],max_depth=config['max_depth'],
            max_features='sqrt',min_samples_leaf=1,bootstrap=True,class_weight=None,random_state=seed,n_jobs=cpu_threads)
    require(model_name=='lightgbm','Unknown tabular model')
    from lightgbm import LGBMClassifier
    return LGBMClassifier(objective='binary',n_estimators=config['n_estimators'],learning_rate=config['lr'],
        num_leaves=31,max_depth=-1,min_child_samples=20,subsample=1.,colsample_bytree=1.,
        class_weight=None,random_state=seed,n_jobs=cpu_threads,verbosity=-1)


def fit_gat_statistics(graph,records):
    """Fit signed-log mean/population SD on unique typed TRAIN neighbors, without labels."""
    by_type={'EOA':[],'CA':[]}
    for record in records:
        batch=torch.load(record['path'],map_location='cpu',weights_only=True)
        for nt in by_type:by_type[nt].append(batch['node_ids'][nt].numpy())
    ids={nt:np.unique(np.concatenate(parts)) for nt,parts in by_type.items()}
    total=torch.zeros(14,dtype=torch.float64);count=sum(len(i) for i in ids.values())
    require(count>0,'Empty GAT TRAIN fitting scope')
    for nt,index in ids.items():
        for start in range(0,len(index),100000):
            values=graph[nt].x[torch.from_numpy(index[start:start+100000])].double()
            values=values.sign()*values.abs().log1p()
            require(bool(torch.isfinite(values).all()),'Nonfinite GAT source14')
            total+=values.sum(0)
    mean=total/count;squared=torch.zeros(14,dtype=torch.float64)
    for nt,index in ids.items():
        for start in range(0,len(index),100000):
            values=graph[nt].x[torch.from_numpy(index[start:start+100000])].double()
            values=values.sign()*values.abs().log1p()
            squared+=(values-mean).square().sum(0)
    scale=(squared/count).sqrt();constant=scale<1e-12;scale[constant]=1.
    return dict(mean=mean.float().tolist(),scale=scale.float().tolist(),
        constant_columns=torch.where(constant)[0].tolist(),unique_train_nodes=count,
        counts_by_type={nt:len(i) for nt,i in ids.items()},labels_used=False,
        fitting_scope='unique (type,id) in TRAIN sampled batches, including unlabeled neighbors; no validation/test batch fitting',
        transform='sign(x)*log1p(abs(x)); population SD; no clipping; frozen float32 buffers',
        computation='chunked float64 mean then centered second pass; chunk summation may differ in rounding from full-tensor audit')


def tabular_predictions(estimator,record):
    """Predict using the saved TRAIN-fitted preprocessing, without fitting any role."""
    with np.load(record['path'],allow_pickle=False) as data:
        values=data['source14'];labels=data['labels'];ids=data['ids']
    require(np.array_equal(estimator.classes_,[0,1]),'Unexpected estimator class order')
    with threadpool_limits(limits=torch.get_num_threads()):scores=estimator.predict_proba(values)[:,1]
    predictions=(scores>.5).astype(np.int64)
    metrics=classification_metrics(labels,scores,predictions)
    rows=pd.DataFrame(dict(node_id=ids,label=labels,score=scores,prediction=predictions))
    return metrics,rows


def run_neural_epoch(study,model,records,role,optimizer=None):
    """Cross-entropy plus the public contrastive loss; supervise sampled roots only."""
    model.train(optimizer is not None);rows=[];loss_sum=0.
    p=study['protocol']
    for record in records:
        batch=torch.load(record['path'],map_location='cpu',weights_only=True)
        batch['batch_name']=Path(record['path']).stem
        y=batch['seed_labels'].to(study['device'])
        with torch.set_grad_enabled(optimizer is not None):
            logits,contrast=forward_batch(model,p['model'],study['graph'],batch,study['job']/'cache'/role/'views',study['device'])
            loss=F.cross_entropy(logits,y)+p['fixed_config'].get('contrast_weight',0.)*contrast
            require(bool(torch.isfinite(logits).all() and torch.isfinite(loss)),'Nonfinite forward/loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True);loss.backward()
                require(all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in model.parameters()),'Nonfinite gradient')
                optimizer.step()
        scores=logits.detach().softmax(dim=1)[:,1].cpu().numpy()
        # Probability threshold is shared with tabular models, including exact float32 ties.
        predictions=(scores>.5).astype(np.int64)
        labels=y.cpu().numpy();loss_sum+=float(loss.detach().cpu())*len(labels)
        rows.extend(dict(node_id=int(i),label=int(y),score=float(s),prediction=int(pred)) for i,y,s,pred in
                    zip(batch['seed_ids'].tolist(),labels,scores,predictions))
    frame=pd.DataFrame(rows)
    require(len(frame)>0 and frame.node_id.is_unique,'Missing/duplicate neural root predictions')
    metrics=classification_metrics(frame.label,frame.score,frame.prediction);metrics['loss']=loss_sum/len(frame)
    return metrics,frame


def fit_one_seed(study,seed,train_records,validation_records):
    """Train once at the fixed configuration; choose neural epochs with validation only."""
    p=study['protocol'];job=study['job'];trial=job/'fits'/f'seed_{seed}';trial.mkdir(parents=True,exist_ok=True)
    completed=trial/'fit.json';identity=json_hash(dict(study=study['identity'],seed=seed,
        input_statistics=study.get('input_statistics')))
    if completed.exists():
        fit=read_json(completed);require(fit['identity']==identity,'Fit identity changed')
        verify_files(trial,fit['files']);return fit
    seed_everything(seed);started=time.perf_counter();captured=[]
    model_name=p['model'];target='EOA' if p['dataset']=='Phish' else 'CA';history=[]
    if model_name in TABULAR:
        estimator=make_estimator(model_name,p['fixed_config'],seed,torch.get_num_threads())
        with np.load(train_records[0]['path'],allow_pickle=False) as data:
            values=data['source14'];labels=data['labels']
        with warnings.catch_warnings(record=True) as captured,threadpool_limits(limits=torch.get_num_threads()):
            warnings.simplefilter('always');estimator.fit(values,labels)
        validation,_=tabular_predictions(estimator,validation_records[0])
        checkpoint=trial/'estimator.joblib';temporary=checkpoint.with_suffix('.partial')
        joblib.dump(estimator,temporary);temporary.replace(checkpoint)
        history=[dict(stage='fit_once',validation_macro_f1=validation['macro_f1'])]
        best_epoch=None;stopped_epoch=None;parameter_count=None
        detail=dict(fitted_features=int(values.shape[1]),training_rows=len(labels),
                    scaler_rows=int(estimator['scale'].n_samples_seen_) if model_name=='lr' else None)
        files=[checkpoint]
    else:
        if model_name=='gat':require(study.get('input_statistics') is not None,'Fit GAT TRAIN statistics before fitting models')
        model=build_model(study['graph'],model_name,p['fixed_config'],target,study.get('input_statistics')).to(study['device'])
        if model_name=='metaifd':model.reset_parameters()
        optimizer=torch.optim.Adam(model.parameters(),lr=p['fixed_config']['lr'],weight_decay=p['weight_decay'])
        best=-1.;stale=0;state=None;best_epoch=None
        for epoch in range(1,p['max_epochs']+1):
            train_metrics,_=run_neural_epoch(study,model,train_records,'train',optimizer)
            validation,_=run_neural_epoch(study,model,validation_records,'validation')
            improved=validation['macro_f1']>=best
            if improved:
                best=validation['macro_f1'];best_epoch=epoch;stale=0
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            history.append(dict(epoch=epoch,train_loss=train_metrics['loss'],validation_macro_f1=validation['macro_f1'],
                                validation_loss=validation['loss'],best_epoch=best_epoch,stale=stale))
            if epoch==1 or epoch%25==0:print(model_name,'seed',seed,'epoch',epoch,'validation Macro-F1',validation['macro_f1'],flush=True)
            if stale>p['patience']:break
        stopped_epoch=epoch;require(state is not None,'No validation checkpoint')
        last=trial/'last_checkpoint.pt'
        atomic_torch_save(dict(state_dict=model.state_dict(),epoch=epoch,identity=identity,diagnostic_only=True),last)
        model.load_state_dict(state,strict=True)
        validation,_=run_neural_epoch(study,model,validation_records,'validation')
        require(abs(validation['macro_f1']-best)<1e-12,'Validation-best checkpoint replay mismatch')
        checkpoint=trial/'checkpoint.pt'
        atomic_torch_save(dict(state_dict=state,best_epoch=best_epoch,identity=identity),checkpoint)
        parameter_count=sum(v.numel() for v in model.parameters());detail={}
        files=[checkpoint,last];del model,optimizer,state;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    pd.DataFrame(history).to_csv(trial/'history.csv',index=False);files.append(trial/'history.csv')
    fit=dict(identity=identity,seed=seed,model=model_name,config=p['fixed_config'],input_width=14,
             validation=validation,best_epoch=best_epoch,stopped_epoch=stopped_epoch,
             parameter_count=parameter_count,training_seconds=time.perf_counter()-started,
             warnings=[str(w.message) for w in captured],detail=detail,test_scored=False,
             checkpoint=checkpoint.name,files=seal_files(trial,files))
    write_json(completed,fit);return fit


def fit_and_lock(study):
    """Save all requested seed fits before allowing any test cache or score."""
    if (study['job']/'model_lock.json').exists():return verify_model_lock(study)
    require(not (study['job']/'cache/test').exists(),'Test cache exists before model lock')
    train_records=prepare_role(study,'train')
    if study['protocol']['model']=='gat':
        statistics=fit_gat_statistics(study['graph'],train_records)
        path=study['job']/'input_statistics.json'
        if path.exists():require(read_json(path)==statistics,'Frozen GAT statistics changed')
        else:write_json(path,statistics)
        study['input_statistics']=statistics
    validation_records=prepare_role(study,'validation')
    fits=[]
    for seed in study['protocol']['model_seeds']:
        fits.append(fit_one_seed(study,seed,train_records,validation_records))
    files=[p for p in (study['job']/'fits').rglob('*') if p.is_file() and not p.name.endswith('.partial')]
    files += [study['job']/'protocol.json',study['job']/'cache/train/manifest.json',study['job']/'cache/validation/manifest.json']
    if (study['job']/'input_statistics.json').exists():files.append(study['job']/'input_statistics.json')
    lock=dict(identity=study['identity'],seeds=study['protocol']['model_seeds'],fits=fits,
              files=seal_files(study['job'],files),test_scored=False)
    write_json(study['job']/'model_lock.json',lock)
    return verify_model_lock(study)


def verify_model_lock(study):
    """Verify the full set of fixed seed estimators/checkpoints and their training caches."""
    path=study['job']/'model_lock.json';require(path.exists(),'Fit and lock every seed before test')
    lock=read_json(path);require(lock['identity']==study['identity'],'Lock identity mismatch')
    require(lock['seeds']==study['protocol']['model_seeds'] and [f['seed'] for f in lock['fits']]==lock['seeds'],
            'Incomplete or duplicated seed fits')
    verify_files(study['job'],lock['files'])
    for role in ('train','validation'):
        root=study['job']/'cache'/role;verify_files(root,read_json(root/'manifest.json')['files'])
    return lock


def evaluate_locked(study):
    """Restore saved fits and predict; this phase never fits models or selects settings."""
    lock=verify_model_lock(study);job=study['job'];result_path=job/'RESULT_MANIFEST.json'
    if result_path.exists():
        result=read_json(result_path)
        require(result['identity']==study['identity'] and result['model_lock_sha256']==sha256(job/'model_lock.json'),'Evaluation identity changed')
        verify_files(job,result['files'])
        return pd.read_csv(job/'metrics.csv')
    records={role:prepare_role(study,role) for role in ('train','validation','test')}
    metrics=[];prediction_dir=job/'predictions';prediction_dir.mkdir(exist_ok=True)
    p=study['protocol'];target='EOA' if p['dataset']=='Phish' else 'CA'
    for fit in lock['fits']:
        trial=job/'fits'/f"seed_{fit['seed']}";verify_files(trial,fit['files'])
        if p['model'] in TABULAR:model=joblib.load(trial/fit['checkpoint'])
        else:
            saved=torch.load(trial/fit['checkpoint'],map_location='cpu',weights_only=True)
            require(saved['identity']==fit['identity'],'Checkpoint identity mismatch')
            model=build_model(study['graph'],p['model'],p['fixed_config'],target).to(study['device'])
            model.load_state_dict(saved['state_dict'],strict=True);model.eval()
        for role in ('train','validation','test'):
            scores,frame=(tabular_predictions(model,records[role][0]) if p['model'] in TABULAR else
                          run_neural_epoch(study,model,records[role],role))
            expected=study['membership'][role].set_index('node_id')
            require(frame.node_id.is_unique and set(frame.node_id)==set(expected.index),'Prediction split coverage mismatch')
            require(np.array_equal(frame.label.to_numpy(),expected.loc[frame.node_id,'label'].to_numpy()),'Prediction label mismatch')
            if role=='validation':require(abs(scores['macro_f1']-fit['validation']['macro_f1'])<1e-12,'Saved validation replay changed')
            frame['address']=expected.loc[frame.node_id,'address'].to_numpy()
            frame.to_csv(prediction_dir/f"seed_{fit['seed']}_{role}.csv",index=False)
            metrics.append(dict(dataset=p['dataset'],model=p['model'],arm='X14',seed=fit['seed'],role=role,**scores))
        del model;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    frame=pd.DataFrame(metrics);frame.to_csv(job/'metrics.csv',index=False)
    keys=['macro_f1','micro_f1','binary_f1','precision','recall','accuracy','fpr','average_precision','roc_auc']
    summary=[]
    for role,part in frame.groupby('role'):
        for key in keys:
            summary.append(dict(dataset=p['dataset'],model=p['model'],role=role,metric=key,
                mean=float(part[key].mean()),sd=float(part[key].std(ddof=1)) if len(part)>1 else None,
                seeds=len(part),unit='fraction; descriptive seed SD on one fixed split'))
    pd.DataFrame(summary).to_csv(job/'summary.csv',index=False)
    files=[job/'metrics.csv',job/'summary.csv']+list(prediction_dir.glob('*.csv'))
    write_json(result_path,dict(identity=study['identity'],model_lock_sha256=sha256(job/'model_lock.json'),
               files=seal_files(job,files),full_benchmark=p['run_mode']=='full',test_used_for_selection=False))
    return frame


def export_results(study,output):
    """Package fitted artifacts and audit files, excluding large sampled topology/view caches."""
    import zipfile
    verify_model_lock(study)
    result=read_json(study['job']/'RESULT_MANIFEST.json');verify_files(study['job'],result['files'])
    output=Path(output)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in study['job'].rglob('*'):
            if not path.is_file() or path.name.endswith('.partial'):continue
            relative=path.relative_to(study['job'])
            if relative.parts[0]=='cache':
                if path.name not in ('manifest.json','index.json'):continue
            archive.write(path,relative.as_posix())
    return output
