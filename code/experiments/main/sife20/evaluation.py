"""Load locked fits, save predictions and summarize split-level paired changes."""
import gc
from pathlib import Path
import zipfile
import joblib
import numpy as np
import pandas as pd
import torch
from .utilities import require,sha256,read_json,write_json,seal_files,verify_files,runtime_information
from .protocol import TABULAR,open_split,verify_study_lock,verify_split_lock
from .cache import prepare_role
from .models import build_classifier
from .training import tabular_predict,neural_epoch


def evaluate_trial(study,split,fit,records):
    """Load one fixed checkpoint; no fitting, threshold search or hyperparameter selection."""
    output=study['output'];trial=output/fit['trial'];verify_files(trial,fit['files'])
    marker=trial/'evaluation.json';prediction_path=trial/'predictions.csv'
    global_hash=sha256(output/'study_lock.json')
    if marker.exists():
        saved=read_json(marker)
        require(saved['study_identity']==study['identity'] and saved['study_lock_sha256']==global_hash,'Evaluation identity changed')
        require(sha256(prediction_path)==saved['predictions_sha256'],'Predictions changed')
        return saved['metrics']
    p=study['protocol'];model_name=p['model'];arm=fit['arm']
    if model_name in TABULAR:model=joblib.load(trial/fit['checkpoint'])
    else:
        checkpoint=torch.load(trial/fit['checkpoint'],map_location='cpu',weights_only=True)
        require(checkpoint['identity']==fit['identity'],'Frozen checkpoint identity mismatch')
        model=build_classifier(model_name,arm,p['config'],study['data']['graph'],p['target'],checkpoint['statistics']).to(study['device'])
        model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
        split['statistics']=checkpoint['statistics']
    rows=[];frames=[]
    for role in ('train','validation','test'):
        metrics,frame=(tabular_predict(model,records[role][0],arm,model_name,p['lr_source14_profile'] or 'signed_log1p') if model_name in TABULAR else
                       neural_epoch(study,split,model,records[role],role,arm))
        expected=split['membership'][role].set_index('node_id')
        require(frame.node_id.is_unique and set(frame.node_id)==set(expected.index),'Prediction role coverage mismatch')
        require(np.array_equal(frame.label.to_numpy(),expected.loc[frame.node_id,'label'].to_numpy()),'Prediction labels changed')
        if role=='validation':require(abs(metrics['macro_f1']-fit['validation_metrics']['macro_f1'])<1e-12,'Saved validation replay mismatch')
        row=dict(dataset=p['dataset'],model=model_name,split_seed=split['seed'],model_seed=fit['model_seed'],arm=arm,role=role,**metrics)
        rows.append(row)
        frame['address']=expected.loc[frame.node_id,'address'].to_numpy()
        for key in ('dataset','model','split_seed','model_seed','arm','role'):frame[key]=row[key]
        frames.append(frame)
    pd.concat(frames,ignore_index=True).to_csv(prediction_path,index=False)
    write_json(marker,dict(study_identity=study['identity'],study_lock_sha256=global_hash,
        predictions_sha256=sha256(prediction_path),metrics=rows))
    del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return rows


def evaluate_all(study):
    """Verify every partition lock before the first TEST prediction; completed runs reuse verified artifacts."""
    lock=verify_study_lock(study);output=study['output']
    require(sha256(output/'protocol.json')==lock['protocol_sha256'],'Frozen protocol changed')
    require(read_json(output/'protocol.json')['runtime']==runtime_information(study['device']),'Runtime changed after fitting')
    if (output/'RESULT_MANIFEST.json').exists():
        result=read_json(output/'RESULT_MANIFEST.json')
        require(result['identity']==study['identity'] and result['study_lock_sha256']==sha256(output/'study_lock.json'),'Result identity changed')
        verify_files(output,result['files'])
        return pd.read_csv(output/'metrics.csv')
    metrics=[]
    for seed in study['protocol']['split_seeds']:
        split=open_split(study,seed);split_lock=verify_split_lock(study,split)
        records={role:prepare_role(study,split,role) for role in ('train','validation','test')}
        for fit in split_lock['fits']:metrics.extend(evaluate_trial(study,split,fit,records))
        pd.DataFrame(metrics).to_csv(output/'evaluation_progress.csv',index=False)
        print('Scored frozen partition',seed,flush=True)
    frame=pd.DataFrame(metrics);frame.to_csv(output/'metrics.csv',index=False)
    summarize(study,frame)
    paths=[output/name for name in ['protocol.json','study_lock.json','metrics.csv','split_means.csv','summary.csv','paired_macro_f1_deltas_pp.csv']]
    paths.extend(output.glob('split_*/seed_*/*/predictions.csv'));paths.extend(output.glob('split_*/seed_*/*/evaluation.json'))
    write_json(output/'RESULT_MANIFEST.json',dict(identity=study['identity'],status='COMPLETE',
        study_lock_sha256=sha256(output/'study_lock.json'),files=seal_files(output,paths),
        full_study=study['protocol']['run_mode']=='full',test_used_for_selection=False))
    return frame


def summarize(study,metrics):
    """Average seeds within each split first; retain every split and paired arm delta."""
    names=['macro_f1','micro_f1','binary_f1','precision','recall','accuracy','average_precision','roc_auc','fpr']
    test=metrics.loc[metrics.role=='test'];p=study['protocol']
    expected={(seed,model_seed,arm) for seed in p['split_seeds'] for model_seed in p['model_seeds'] for arm in p['arms']}
    require(set(zip(test.split_seed,test.model_seed,test.arm))==expected and len(test)==len(expected),'Incomplete test coverage')
    splits=test.groupby(['dataset','model','arm','split_seed'])[names].mean().reset_index()
    splits.to_csv(study['output']/'split_means.csv',index=False);rows=[]
    for arm,frame in splits.groupby('arm'):
        for key in names:
            values=frame[key].to_numpy()
            rows.append(dict(dataset=p['dataset'],model=p['model'],arm=arm,metric=key,
                mean_percent=float(values.mean()*100),sd_across_split_means_pp=float(values.std(ddof=1)*100) if len(values)>1 else None,
                split_count=len(values),model_seeds_per_split=len(p['model_seeds']),
                inference='descriptive repeated holdout; overlapping corpus; not100independent datasets'))
    summary=pd.DataFrame(rows);summary.to_csv(study['output']/'summary.csv',index=False)
    pivot=splits.pivot(index='split_seed',columns='arm',values='macro_f1')
    deltas=pivot.subtract(pivot['X14'],axis=0).drop(columns='X14')*100
    deltas.to_csv(study['output']/'paired_macro_f1_deltas_pp.csv')
    return summary


def export_results(study,output,allow_partial=False,include_caches=False):
    """Export partial progress or complete audit evidence; full snapshots are available for exact resume."""
    root=study['output'];complete=(root/'RESULT_MANIFEST.json').exists()
    require(complete or allow_partial,'Evaluation is incomplete; explicitly request a PARTIAL progress export')
    if complete:
        verify_study_lock(study);verify_files(root,read_json(root/'RESULT_MANIFEST.json')['files'])
    output=Path(output)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in root.rglob('*'):
            if not path.is_file() or path.name.endswith('.partial'):continue
            if path.resolve()==output.resolve():continue
            relative=path.relative_to(root)
            if not include_caches and 'cache' in relative.parts and path.name not in ('manifest.json','index.json'):continue
            archive.write(path,'results/'+study['protocol']['run_mode']+'/'+relative.as_posix())
        data_output=study['data'].get('data_output')
        if data_output:
            for name in ('roots.npz','data_manifest.json'):
                path=Path(data_output)/name
                if path.exists():archive.write(path,'data/'+name)
        archive.writestr('EXPORT_SCOPE.json',__import__('json').dumps(dict(complete=complete,caches_included=include_caches,
            resume='restore full WORK_ROOT including sampled cache files; small audit ZIP alone omits required large cached draws',
            raw_graph_and_ponzi_sidecars_included=False),indent=2))
    return output
