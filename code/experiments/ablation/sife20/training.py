"""TRAIN/validation fitting only; no test inference is reachable from this module."""
import gc
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
from .features import select_inputs
from .tabular_inputs import tabular_input_frame,check_estimator_columns
from .models import ARM_WIDTHS,build_classifier,forward_classifier
from .metrics import classification_metrics
from .protocol import TABULAR,open_split,verify_split_lock,verify_study_lock
from .cache import prepare_role,fit_training_statistics
from .utilities import require,sha256,json_hash,write_json,read_json,seed_everything,atomic_torch_save,seal_files,verify_files


def make_estimator(model_name,config,seed,threads):
    if model_name=='lr':
        return Pipeline([('scale',StandardScaler()),('classifier',LogisticRegression(C=config['C'],solver=config['solver'],
            max_iter=config['max_iter'],class_weight=None,random_state=seed))])
    if model_name=='rf':
        return RandomForestClassifier(n_estimators=config['n_estimators'],max_depth=config['max_depth'],
            max_features='sqrt',min_samples_leaf=1,class_weight=None,bootstrap=True,random_state=seed,n_jobs=threads)
    require(model_name=='lightgbm','Unknown tabular model')
    from lightgbm import LGBMClassifier
    return LGBMClassifier(objective='binary',n_estimators=config['n_estimators'],learning_rate=config['lr'],
        num_leaves=config['num_leaves'],max_depth=-1,min_child_samples=20,subsample=1.,colsample_bytree=1.,
        class_weight=None,random_state=seed,n_jobs=threads,verbosity=-1)


def tabular_predict(estimator,record,arm,model_name,lr_profile='signed_log1p'):
    """Predict with a saved pipeline; Pure36 passes no source14 to the input selector."""
    with np.load(record['path'],allow_pickle=False) as stored:
        x=tabular_input_frame(None if arm=='Pure36' else stored['source14'],stored['sife44'],arm,model_name,lr_profile)
        ids=stored['ids'].copy();labels=stored['labels'].copy()
    require(np.array_equal(estimator.classes_,[0,1]),'Unexpected class order')
    check_estimator_columns(estimator,x)
    with threadpool_limits(limits=torch.get_num_threads()):
        scores=estimator.predict_proba(x,validate_features=True)[:,1] if model_name=='lightgbm' else estimator.predict_proba(x)[:,1]
    predictions=(scores>.5).astype(np.int64)
    return classification_metrics(labels,scores,predictions),pd.DataFrame(dict(node_id=ids,label=labels,score=scores,prediction=predictions))


def neural_epoch(study,split,model,records,role,arm,optimizer=None):
    """Supervise seed/root labels only; losses use common sampled inputs for every arm."""
    model.train(optimizer is not None);rows=[];loss_sum=0.
    for record in records:
        batch=torch.load(record['path'],map_location='cpu',weights_only=True)
        batch['batch_name']=Path(record['path']).stem;y=batch['seed_labels'].to(study['device'])
        with torch.set_grad_enabled(optimizer is not None):
            logits,contrast=forward_classifier(model,study,split,arm,batch,role)
            loss=F.cross_entropy(logits,y)+study['protocol']['config'].get('contrast_weight',0.)*contrast
            require(bool(torch.isfinite(logits).all() and torch.isfinite(loss)),'Nonfinite logits/loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True);loss.backward()
                require(all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in model.parameters()),'Nonfinite gradients')
                optimizer.step()
        scores=logits.detach().softmax(1)[:,1].cpu().numpy();labels=y.cpu().numpy()
        predictions=(scores>.5).astype(np.int64);loss_sum+=float(loss.detach().cpu())*len(labels)
        rows.extend(dict(node_id=int(i),label=int(label),score=float(score),prediction=int(pred)) for i,label,score,pred in
                    zip(batch['seed_ids'].tolist(),labels,scores,predictions))
    frame=pd.DataFrame(rows);require(len(frame)>0 and frame.node_id.is_unique,'Missing/duplicate root predictions')
    metrics=classification_metrics(frame.label,frame.score,frame.prediction);metrics['loss']=loss_sum/len(frame)
    return metrics,frame


def fit_arm_seed(study,split,arm,seed,train_records,validation_records):
    """Fit one arm/seed at the fixed config; replay validation-best before saving the completed marker."""
    p=study['protocol'];require(arm in p['arms'] and seed in p['model_seeds'],'Unsupported arm or seed')
    trial=split['folder']/f'seed_{seed}'/arm;trial.mkdir(parents=True,exist_ok=True)
    identity=json_hash(dict(split=split['identity'],arm=arm,seed=seed,statistics=split.get('statistics')))
    path=trial/'fit.json'
    if path.exists():
        fit=read_json(path);require(fit['identity']==identity,'Fit belongs to another data/config/preprocessing context')
        verify_files(trial,fit['files']);return fit
    seed_everything(seed);started=time.perf_counter();history=[];captured=[]
    model_name=p['model'];parameter_count=None;detail={};best_epoch=None;stopped_epoch=None
    if model_name in TABULAR:
        with np.load(train_records[0]['path'],allow_pickle=False) as stored:
            x=tabular_input_frame(None if arm=='Pure36' else stored['source14'],stored['sife44'],arm,model_name,p['lr_source14_profile'] or 'signed_log1p')
            labels=stored['labels']
        estimator=make_estimator(model_name,p['config'],seed,torch.get_num_threads())
        with warnings.catch_warnings(record=True) as captured,threadpool_limits(limits=torch.get_num_threads()):
            warnings.simplefilter('always');estimator.fit(x,labels)
        validation,_=tabular_predict(estimator,validation_records[0],arm,model_name,p['lr_source14_profile'] or 'signed_log1p')
        training_metrics,training_predictions=tabular_predict(estimator,train_records[0],arm,model_name,p['lr_source14_profile'] or 'signed_log1p')
        single_class=bool(training_predictions.prediction.nunique()<2)
        checkpoint=trial/'model.joblib';temporary=checkpoint.with_suffix('.partial')
        joblib.dump(estimator,temporary);temporary.replace(checkpoint)
        detail=dict(train_roots=len(labels),input_width=x.shape[1],
            scaler_fit_rows=int(estimator['scale'].n_samples_seen_) if model_name=='lr' else None,
            n_iter=np.asarray(estimator['classifier'].n_iter_).tolist() if model_name=='lr' else None,
            lr_source14_profile=p['lr_source14_profile'],feature_names=list(x.columns),
            single_class_train_prediction=single_class,train_metrics=training_metrics,
            train_predicted_positives=int(training_predictions.prediction.sum()),
            validation_predicted_positives=int(validation['tp']+validation['fp']),
            estimator_input_contract='tabular_input_frame(profile) first, then the saved scaler/classifier; do not feed raw values directly to a signed-log fit')
        if single_class:
            print('DIAGNOSTIC:',p['dataset'],model_name,split['seed'],arm,seed,
                  'TRAIN predictions contain one class; recorded in fit.json. Config/profile/threshold are not changed automatically.',flush=True)
        history=[dict(stage='one_fixed_budget_fit',validation_macro_f1=validation['macro_f1'])];files=[checkpoint]
        del estimator
    else:
        require(split.get('statistics') is not None,'TRAIN preprocessing must be frozen first')
        model=build_classifier(model_name,arm,p['config'],study['data']['graph'],p['target'],split['statistics']).to(study['device'])
        if model_name=='metaifd':model.reset_parameters()
        optimizer=torch.optim.Adam(model.parameters(),lr=p['config']['lr'],weight_decay=p['weight_decay'])
        best=-1.;stale=0;state=None
        for epoch in range(1,p['max_epochs']+1):
            training,_=neural_epoch(study,split,model,train_records,'train',arm,optimizer)
            validation,_=neural_epoch(study,split,model,validation_records,'validation',arm)
            if validation['macro_f1']>=best:
                best=validation['macro_f1'];best_epoch=epoch;stale=0
                state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            else:stale+=1
            history.append(dict(epoch=epoch,train_loss=training['loss'],validation_loss=validation['loss'],
                                validation_macro_f1=validation['macro_f1'],best_epoch=best_epoch,stale=stale))
            if epoch==1 or epoch%25==0:
                print(p['dataset'],model_name,'split',split['seed'],arm,'seed',seed,'epoch',epoch,'val',validation['macro_f1'],flush=True)
            if stale>p['patience']:break
        require(state is not None,'No validation checkpoint selected');stopped_epoch=epoch
        last=trial/'last_checkpoint.pt'
        atomic_torch_save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                               identity=identity,last_epoch=epoch,diagnostic_only=True),last)
        model.load_state_dict(state,strict=True)
        validation,_=neural_epoch(study,split,model,validation_records,'validation',arm)
        require(abs(validation['macro_f1']-best)<1e-12,'Validation-best checkpoint replay mismatch')
        checkpoint=trial/'checkpoint.pt'
        atomic_torch_save(dict(state_dict=state,identity=identity,best_epoch=best_epoch,
                               arm=arm,statistics=split['statistics']),checkpoint)
        parameter_count=sum(v.numel() for v in model.parameters());files=[checkpoint,last]
        del model,optimizer,state;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    pd.DataFrame(history).to_csv(trial/'history.csv',index=False);files.append(trial/'history.csv')
    fit=dict(identity=identity,study_identity=study['identity'],split_identity=split['identity'],split_seed=split['seed'],
        model_seed=seed,arm=arm,n_features=ARM_WIDTHS[arm],config=p['config'],validation_metrics=validation,
        best_epoch=best_epoch,stopped_epoch=stopped_epoch,parameter_count=parameter_count,detail=detail,
        fit_seconds=time.perf_counter()-started,warnings=[str(w.message) for w in captured],
        checkpoint=checkpoint.name,trial=trial.relative_to(study['output']).as_posix(),files=seal_files(trial,files),test_scored=False)
    write_json(path,fit);return fit


class SessionBudgetReached(RuntimeError):
    """A resumable stop between fits, not a completed benchmark."""
    pass


def fit_split(study,split,deadline=None):
    if (split['folder']/'lock.json').exists():return verify_split_lock(study,split)
    require(not (study['output']/'study_lock.json').exists(),'Whole-study fitting is closed after lock')
    require(not (split['folder']/'cache/test').exists(),'Test cache appeared before fit lock')
    train=prepare_role(study,split,'train')
    if study['protocol']['model'] not in TABULAR:fit_training_statistics(study,split)
    validation=prepare_role(study,split,'validation');fits=[]
    for seed in study['protocol']['model_seeds']:
        for arm in study['protocol']['arms']:
            if deadline is not None and time.monotonic()>=deadline:
                raise SessionBudgetReached('Session budget reached between fits. Keep WORK_ROOT and rerun unchanged settings to resume. No TEST was scored.')
            fits.append(fit_arm_seed(study,split,arm,seed,train,validation))
            pd.DataFrame([dict(split_seed=r['split_seed'],model_seed=r['model_seed'],arm=r['arm'],
                validation_macro_f1=r['validation_metrics']['macro_f1'],fit_seconds=r['fit_seconds']) for r in fits]).to_csv(split['folder']/'fit_progress.csv',index=False)
    files=[split['folder']/'membership.json',split['folder']/'cache/train/manifest.json',split['folder']/'cache/validation/manifest.json']
    for fit in fits:
        trial=study['output']/fit['trial'];files.extend(trial/name for name in fit['files']);files.append(trial/'fit.json')
    if (split['folder']/'input_statistics.json').exists():files.append(split['folder']/'input_statistics.json')
    lock=dict(identity=split['identity'],fits=fits,files=seal_files(split['folder'],files),test_scored_at_lock=False)
    write_json(split['folder']/'lock.json',lock)
    print('Frozen partition',split['seed'],'fits',len(fits),flush=True)
    return verify_split_lock(study,split)


def fit_all(study,deadline=None):
    """Complete every requested split/arm/seed before enabling the test evaluation cell."""
    if (study['output']/'study_lock.json').exists():return verify_study_lock(study)
    hashes={}
    for seed in study['protocol']['split_seeds']:
        split=open_split(study,seed);fit_split(study,split,deadline)
        hashes[str(seed)]=sha256(split['folder']/'lock.json')
    lock=dict(identity=study['identity'],split_lock_hashes=hashes,protocol_sha256=sha256(study['output']/'protocol.json'),test_scored_at_lock=False)
    write_json(study['output']/'study_lock.json',lock)
    return verify_study_lock(study)
