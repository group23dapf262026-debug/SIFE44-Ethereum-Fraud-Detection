"""X14 baseline only: one released split, five fixed fits, locked test evaluation."""
import hashlib
from pathlib import Path
import time
import warnings
import zipfile
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .utilities import require,sha256,json_hash,read_json,write_json,seal_files,verify_files,runtime_information
from .metrics import classification_metrics

SOURCE14_NAMES=('bal_c','V_inv_c','V_return_c','N_inv_c','N_return_c','V_meaninv_c','V_meanreturn_c',
                'bal_t','V_inv_t','V_return_t','N_inv_t','N_return_t','V_meaninv_t','V_meanreturn_t')


def notebook_source_signature():
    """Record literal scientific definitions executed in this notebook."""
    history=get_ipython().history_manager.input_hist_raw;latest={}
    for source in history:
        first=source.splitlines()[0] if source.splitlines() else ''
        if first.startswith('# BASELINE_SOURCE_CELL: '):
            latest[first.split(': ',1)[1]]=hashlib.sha256(source.encode()).hexdigest()
    require(set(DEFINITION_KEYS)<=set(latest),'Run all definition cells first')
    return {key:latest[key] for key in DEFINITION_KEYS}


def fixed_configuration(model,profile):
    """Keep the historical reference settings; no preprocessing/loss substitution."""
    require(model in ('lr','lightgbm') and profile=='reference','Only the unchanged reference baseline is supported')
    if model=='lr':
        return dict(C=10.,solver='liblinear',max_iter=5000,class_weight=None)
    return dict(n_estimators=300,learning_rate=.01,num_leaves=31,class_weight=None)


def input_frame(source14,model,profile):
    """Exactly14 unchanged source columns, represented by stable names and order."""
    values=np.asarray(source14,dtype=np.float64).copy()
    require(values.ndim==2 and values.shape[1]==14,'X14 baseline requires exactly14source columns')
    require(profile=='reference','Reference baseline must use raw X14')
    require(np.isfinite(values).all(),'Nonfinite baseline inputs')
    return pd.DataFrame(values,columns=SOURCE14_NAMES)


def open_study(graph,membership,data_manifest,output,dataset,model,profile,source_signature,cpu_threads=2):
    """Freeze the released split, fixed configuration, preprocessing and five seeds."""
    torch.set_num_threads(cpu_threads);config=fixed_configuration(model,profile)
    protocol=dict(study='released_x14_tabular_fix_v3',dataset=dataset,model=model,arm='X14',width=14,
        profile=profile,fixed_config=config,model_seeds=[12,13,14,15,16],
        split='unchanged released train/val/test masks; no resplitting or cohort filtering',
        hyperparameter_search=False,decision='score1>0.5; exact tie class0',
        preprocessing='raw X14 then TRAIN-only StandardScaler' if model=='lr' else 'raw X14; no scaler',
        class_weight_scope='unweighted',
        paper_status=('historical accepted-study raw LR; LR absent from original Meta-IFD TableIV' if model=='lr' else
                      'accepted-study fixed LightGBM; original paper reports search space, not recovered per-dataset winning settings'),
        test_policy='fit and lock all five seeds before TEST scoring; evaluate saved estimators only',
        source_columns=list(SOURCE14_NAMES),setting='static released benchmark; known Phish address conflict retained and disclosed',
        development_provenance='prior corpus results seen; no claim of a previously unseen test corpus',
        score_interpretation='uncalibrated classifier score')
    identity=dict(protocol=protocol,data_manifest=data_manifest,source_signature=source_signature,
        memberships={role:table.to_dict('records') for role,table in membership.items()},runtime=runtime_information('cpu'))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():require(read_json(output/'protocol.json')==identity,'Run identity changed; use a new output folder')
    else:write_json(output/'protocol.json',identity)
    for role,table in membership.items():table.to_csv(output/(role+'_membership.csv'),index=False)
    return dict(graph=graph,membership=membership,protocol=protocol,identity=json_hash(identity),output=output)


def prepare_role(study,role):
    """Cache unchanged source14 at released root IDs; TEST requires the complete model lock."""
    require(role in ('train','validation','test'),'Unknown role')
    if role=='test':verify_model_lock(study)
    path=study['output']/'cache'/role/'roots.npz';marker=path.with_suffix('.json')
    if marker.exists():
        record=read_json(marker);require(record['identity']==study['identity'] and record['sha256']==sha256(path),'Root cache changed')
    else:
        table=study['membership'][role];target='EOA' if study['protocol']['dataset']=='Phish' else 'CA'
        path.parent.mkdir(parents=True,exist_ok=True)
        values=study['graph'][target].x[torch.tensor(table.node_id.to_numpy(),dtype=torch.long)].numpy()
        np.savez_compressed(path,ids=table.node_id.to_numpy(),labels=table.label.to_numpy(),source14=values)
        write_json(marker,dict(identity=study['identity'],sha256=sha256(path),rows=len(table)))
    return path


def make_estimator(model,config,seed,threads):
    """Standard portable estimators; no custom pickled transforms or hidden search."""
    if model=='lr':return Pipeline([('scale',StandardScaler()),('classifier',LogisticRegression(random_state=seed,**config))])
    from lightgbm import LGBMClassifier
    return LGBMClassifier(objective='binary',random_state=seed,n_jobs=threads,verbosity=-1,**config)


def predict_roots(estimator,path,model,profile):
    """Apply the declared input transform and reject changed feature names/order."""
    with np.load(path,allow_pickle=False) as values:
        x=input_frame(values['source14'],model,profile);ids=values['ids'].copy();labels=values['labels'].copy()
    require(tuple(estimator.feature_names_in_)==tuple(x.columns),'Feature column names/order changed')
    require(np.array_equal(estimator.classes_,[0,1]),'Unexpected class order')
    with threadpool_limits(limits=torch.get_num_threads()):
        score=estimator.predict_proba(x,validate_features=True)[:,1] if model=='lightgbm' else estimator.predict_proba(x)[:,1]
    prediction=(score>.5).astype(int)
    return classification_metrics(labels,score,prediction),pd.DataFrame(dict(node_id=ids,label=labels,score=score,prediction=prediction))


def fit_and_lock(study):
    """Fit TRAIN, record validation/diagnostics, save all five fits before enabling TEST."""
    output=study['output'];p=study['protocol']
    if (output/'model_lock.json').exists():return verify_model_lock(study)
    require(not (output/'cache/test').exists(),'TEST cache present before model lock')
    train=prepare_role(study,'train');validation=prepare_role(study,'validation');fits=[]
    with np.load(train) as record:
        x=input_frame(record['source14'],p['model'],p['profile']);labels=record['labels'].copy()
    counts=np.bincount(labels,minlength=2)
    for seed in p['model_seeds']:
        trial=output/'fits'/f'seed_{seed}';trial.mkdir(parents=True,exist_ok=True)
        marker=trial/'fit.json'
        if marker.exists():
            fit=read_json(marker);require(fit['identity']==study['identity'] and fit['seed']==seed,'Fit identity changed')
            verify_files(trial,fit['files']);fits.append(fit);continue
        estimator=make_estimator(p['model'],p['fixed_config'],seed,torch.get_num_threads());started=time.perf_counter()
        with warnings.catch_warnings(record=True) as observed,threadpool_limits(limits=torch.get_num_threads()):
            warnings.simplefilter('always');estimator.fit(x,labels)
        train_metrics,train_predictions=predict_roots(estimator,train,p['model'],p['profile'])
        val_metrics,_=predict_roots(estimator,validation,p['model'],p['profile'])
        checkpoint=trial/'estimator.joblib';temp=checkpoint.with_suffix('.partial');joblib.dump(estimator,temp);temp.replace(checkpoint)
        detail=dict(train_class_counts=counts.tolist(),resolved_train_class_weights={str(c):float(len(labels)/(2*counts[c]))
            if p['fixed_config']['class_weight']=='balanced' else 1. for c in (0,1)},
            train_predicted_positives=int(train_predictions.prediction.sum()),train_single_class=bool(train_predictions.prediction.nunique()<2),
            validation_predicted_positives=val_metrics['tp']+val_metrics['fp'],
            n_iter=estimator['classifier'].n_iter_.tolist() if p['model']=='lr' else None,
            scaler_fit_rows=int(estimator['scale'].n_samples_seen_) if p['model']=='lr' else None,
            input_contract='call input_frame(raw_source14,model,profile) before saved estimator prediction',
            feature_names=list(x.columns))
        fit=dict(identity=study['identity'],seed=seed,config=p['fixed_config'],profile=p['profile'],detail=detail,
            training=train_metrics,validation=val_metrics,training_seconds=time.perf_counter()-started,
            warnings=[str(item.message) for item in observed],files=seal_files(trial,[checkpoint]))
        write_json(marker,fit);fits.append(fit)
        print('Fitted baseline',p['dataset'],p['model'],'seed',seed,'validation Macro-F1',val_metrics['macro_f1'],
              '| TRAIN predicted positives:',detail['train_predicted_positives'],flush=True)
    files=[output/'protocol.json']+list(output.glob('*_membership.csv'))+list((output/'fits').glob('seed_*/*'))+list((output/'cache').rglob('*'))
    files=[path for path in files if path.is_file()]
    lock=dict(identity=study['identity'],fits=fits,files=seal_files(output,files),test_scored_at_lock=False)
    write_json(output/'model_lock.json',lock);return verify_model_lock(study)


def verify_model_lock(study):
    """Check fixed seed coverage, input/fit hashes and unchanged runtime before TEST."""
    path=study['output']/'model_lock.json';require(path.exists(),'Fit and lock all five seeds before TEST')
    lock=read_json(path);require(lock['identity']==study['identity'],'Model lock identity changed')
    require(len(lock['fits'])==5 and {f['seed'] for f in lock['fits']}==set(study['protocol']['model_seeds']),'Missing seed')
    verify_files(study['output'],lock['files'])
    require(read_json(study['output']/'protocol.json')['runtime']==runtime_information('cpu'),'Runtime changed after fitting')
    return lock


def evaluate_locked(study):
    """Restore locked estimators, replay validation, score the released TEST once."""
    output=study['output'];lock=verify_model_lock(study);marker=output/'RESULT_MANIFEST.json'
    if marker.exists():
        record=read_json(marker);require(record['identity']==study['identity'] and record['lock_sha256']==sha256(output/'model_lock.json'),'Evaluation identity changed')
        verify_files(output,record['files']);return pd.read_csv(output/'metrics.csv')
    records={role:prepare_role(study,role) for role in ('train','validation','test')};p=study['protocol'];rows=[]
    folder=output/'predictions';folder.mkdir(exist_ok=True)
    for fit in lock['fits']:
        estimator=joblib.load(output/'fits'/f"seed_{fit['seed']}"/'estimator.joblib')
        for role,path in records.items():
            metrics,frame=predict_roots(estimator,path,p['model'],p['profile'])
            expected=study['membership'][role].set_index('node_id')
            require(frame.node_id.is_unique and set(frame.node_id)==set(expected.index),'Prediction coverage changed')
            require(np.array_equal(frame.label,expected.loc[frame.node_id,'label']),'Prediction labels changed')
            if role=='validation':require(abs(metrics['macro_f1']-fit['validation']['macro_f1'])<1e-12,'Saved validation replay changed')
            frame['address']=expected.loc[frame.node_id,'address'].to_numpy();frame.to_csv(folder/f"seed_{fit['seed']}_{role}.csv",index=False)
            rows.append(dict(dataset=p['dataset'],model=p['model'],profile=p['profile'],arm='X14',seed=fit['seed'],role=role,**metrics))
    frame=pd.DataFrame(rows);frame.to_csv(output/'metrics.csv',index=False)
    keys=['macro_f1','micro_f1','binary_f1','precision','recall','accuracy','fpr','average_precision','roc_auc']
    summary=[]
    for role,part in frame.groupby('role'):
        for metric in keys:summary.append(dict(dataset=p['dataset'],model=p['model'],profile=p['profile'],role=role,metric=metric,
            mean_percent=float(part[metric].mean()*100),sd_across_model_seeds_pp=float(part[metric].std(ddof=1)*100),seeds=5))
    pd.DataFrame(summary).to_csv(output/'summary.csv',index=False)
    files=[output/'metrics.csv',output/'summary.csv']+list(folder.glob('*.csv'))
    write_json(marker,dict(identity=study['identity'],lock_sha256=sha256(output/'model_lock.json'),files=seal_files(output,files),
                          status='COMPLETE',test_used_for_selection=False))
    return frame


def export_results(study,path):
    """Export the X14-only baseline, including small root caches and diagnostics."""
    verify_model_lock(study);verify_files(study['output'],read_json(study['output']/'RESULT_MANIFEST.json')['files'])
    path=Path(path)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in study['output'].rglob('*'):
            if entry.is_file() and entry.resolve()!=path.resolve() and not entry.name.endswith('.partial'):
                archive.write(entry,entry.relative_to(study['output']).as_posix())
    return path
