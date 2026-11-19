"""Seed-then-partition summaries and assumption-dependent corrected resampling sensitivity."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
MODELS=('metaifd','gat','sage','lightgbm','rf')
DATASETS=('Phish','Ponzi')
PAIRS={
 'direct': [('X14_SIFE44','X14_UFE12'),('X14_Pure36','X14_UFE12'),
            ('X14_UFE12_SIFE44','X14_UFE12'),('X14_UFE12_Pure36','X14_UFE12')],
 'shuffle': [('X14_SIFE44','X14'),('X14_SHUFFLED_SIFE44','X14'),('X14_SIFE44','X14_SHUFFLED_SIFE44')],
 'frozen': [('X14_Pure36','X14'),('X14_SIFE44','X14')]
}
METRICS=('macro_f1','average_precision','recall','precision','fpr','binary_f1','roc_auc','accuracy','micro_f1')


def json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value,message):
    if not value: raise ValueError(message)


def holm(values):
    values=np.asarray(values,float)
    order=np.argsort(values,kind='stable')
    result=np.empty(len(values))
    result[order]=np.minimum(1.,np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    return result


def corrected(deltas,ratio):
    """Pointwise t interval; Holm affects p-values, not interval endpoints."""
    values=np.asarray(deltas,float)
    require(len(values)==20 and np.isfinite(values).all(),'Exactly twenty finite partition differences required')
    mean=float(values.mean());sd=float(values.std(ddof=1))
    se=float(np.sqrt((1/20+ratio)*sd**2))
    p=float(2*stats.t.sf(abs(mean/se),19)) if se>0 else float(mean==0)
    width=float(stats.t.ppf(.975,19)*se)
    return dict(mean_delta_pp=mean,sd_delta_pp=sd,approximate_corrected_se_pp=se,
                pointwise95_low_pp=mean-width,pointwise95_high_pp=mean+width,two_sided_p=p,
                correction_ratio_test_train=ratio,df=19,wins=int((values>0).sum()),
                ties=int((values==0).sum()),losses=int((values<0).sum()))


def analyze(frame,experiment,ratios,output):
    require(set(frame.dataset)==set(DATASETS) and set(frame.model)==set(MODELS),'All ten dataset/model tasks required')
    keys=['dataset','model','arm','split_seed','model_seed']
    test=frame.loc[frame.role=='test'].copy()
    require(not test.duplicated(keys).any(),'Duplicated fits; do not pool reruns')
    arms={a for pair in PAIRS[experiment] for a in pair}
    if experiment=='frozen':test=test.loc[test.arm.isin(arms)]
    expected={(d,m,a,s,t) for d in DATASETS for m in MODELS for a in arms for s in range(2001,2021) for t in range(12,17)}
    require(set(test[keys].itertuples(index=False,name=None))==expected,'Incomplete or unexpected arm/split/seed coverage')
    require(np.isfinite(test[list(METRICS)].to_numpy()).all(),'Nonfinite metric')
    require(((test[list(METRICS)]>=0)&(test[list(METRICS)]<=1)).all().all(),'Metrics must be proportions')
    split=test.groupby(keys[:-1])[list(METRICS)].mean().reset_index()
    summaries=[];comparisons=[];differences=[]
    for (ds,model,arm),group in split.groupby(['dataset','model','arm']):
        for metric in METRICS:
            v=100*group[metric]
            summaries.append(dict(dataset=ds,model=model,arm=arm,metric=metric,mean_percent=v.mean(),sd_across_partition_means_pp=v.std(ddof=1)))
    for (ds,model),group in split.groupby(['dataset','model']):
        pivot=group.pivot(index='split_seed',columns='arm',values='macro_f1').sort_index()
        for candidate,control in PAIRS[experiment]:
            delta=100*(pivot[candidate]-pivot[control])
            comparisons.append(dict(dataset=ds,model=model,candidate=candidate,control=control,
                                    **corrected(delta.to_numpy(),ratios[ds])))
            differences.extend(dict(dataset=ds,model=model,candidate=candidate,control=control,split_seed=int(s),delta_pp=float(v)) for s,v in delta.items())
    inference=pd.DataFrame(comparisons)
    inference['holm_p']=holm(inference.two_sided_p)
    inference['holm_family_size']=len(inference)
    inference['scope']='assumption-dependent sensitivity; overlapping graph holdouts and nonadditive F1 are not solved'
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,table in [('split_means',split),('summary',pd.DataFrame(summaries)),('paired_deltas',pd.DataFrame(differences)),('corrected_sensitivity',inference)]:
        table.to_csv(output/f'{name}.csv',index=False)
    return inference


def replay_frozen(repo,output):
    frames=[pd.read_csv(p,float_precision='round_trip') for p in sorted((repo/'results/main/runs').glob('*/metrics.csv'))]
    require(len(frames)==10,'Expected ten frozen main-study exports')
    ratios={}
    for ds in DATASETS:
        membership=json.loads((repo/'results/cohorts'/ds.lower()/'split_2001.json').read_text())
        ratios[ds]=len(membership['test'])/len(membership['train'])
    actual=analyze(pd.concat(frames,ignore_index=True),'frozen',ratios,output)
    recorded=pd.read_csv(repo/'results/activity_matching/ASSUMPTION_DEPENDENT_INFERENCE_SENSITIVITY.csv',float_precision='round_trip')
    a=actual.rename(columns={'candidate':'arm','holm_p':'holm_p_20_comparisons'}).set_index(['dataset','model','arm']).sort_index()
    b=recorded.set_index(['dataset','model','arm']).sort_index()
    require(a.index.equals(b.index),'Frozen inference coverage differs')
    columns=['mean_delta_pp','approximate_corrected_se_pp','pointwise95_low_pp','pointwise95_high_pp','two_sided_p','holm_p_20_comparisons']
    np.testing.assert_allclose(a[columns],b[columns],rtol=0,atol=1e-10)
    report=dict(status='PASS',comparisons=20,holm_family='both datasets x five models x two augmentation arms',
                seed_averaging='five seeds within partition before twenty-partition inference',
                pointwise_intervals_not_multiplicity_adjusted=True,
                metrics_sha256={p.parent.name:sha(p) for p in sorted((repo/'results/main/runs').glob('*/metrics.csv'))},
                recorded_inference_sha256=sha(repo/'results/activity_matching/ASSUMPTION_DEPENDENT_INFERENCE_SENSITIVITY.csv'),
                matched_to_recorded_tolerance=1e-10)
    (Path(output)/'REPLAY_CHECK.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(actual[['dataset','model','candidate','mean_delta_pp','holm_p']].to_string(index=False))


def collect_followup(folder,experiment,output):
    """Read extracted complete exports; check scientific identities before combining tasks."""
    frames=[];seen=set();evidence=[]
    expected_arms={a for pair in PAIRS[experiment] for a in pair}
    refs={ds:json.loads((ROOT/'assets'/f'release_reference_{ds}.json').read_text()) for ds in DATASETS}
    for path in sorted(Path(folder).rglob('RESULT_MANIFEST.json')):
        root=path.parent
        identity=json.loads((root/'protocol.json').read_text())
        p=identity['protocol']
        if p.get('experiment')!=experiment:continue
        require(p['study']=='sife_revision_v1_20261112' and p['run_mode']=='full','Only full follow-up studies are accepted')
        plan=json.loads((ROOT/'configs/ANALYSIS_PLAN.json').read_text())
        require(p['dataset'] in DATASETS and p['model'] in MODELS,'Unknown dataset/model')
        require(p['config']==plan['fixed_configurations'][p['model']],'Fixed model configuration differs')
        require(p['decision']==plan['decision'],'Threshold rule differs')
        require(p['comparisons']==[list(pair) for pair in PAIRS[experiment]],'Comparison plan differs')
        require(p['feature_version']=='sife_revision_exact_ufe12_v1_20261112','Feature version differs')
        require(not p['hyperparameter_search'],'Unexpected hyperparameter search')
        require(set(p['arms'])==expected_arms,'Unexpected arm set')
        require(p['arms']=={arm:plan['arm_widths'][arm] for arm in expected_arms},'Input widths differ')
        require(p['split_seeds']==list(range(2001,2021)) and p['model_seeds']==list(range(12,17)),'Unexpected seed plan')
        result=json.loads(path.read_text())
        require(result['status']=='COMPLETE' and result['full_study'] and result['identity']==json_hash(identity),'Result identity/completeness mismatch')
        require(not result['test_used_for_selection'],'TEST selection flag is set')
        for relative,expected in result['files'].items():
            require(sha(root/relative)==expected,f'Changed result artifact: {relative}')
        require(sha(root/'study_lock.json')==result['study_lock_sha256'],'Changed study lock')
        lock=json.loads((root/'study_lock.json').read_text())
        require(lock['identity']==result['identity'] and not lock['test_scored_at_lock'],'Study was not locked before TEST')
        data=identity['data_manifest']['inputs']
        ref=refs[p['dataset']]
        require(data['graph_sha256']==ref['graph_sha256'],'Graph bytes differ')
        require(data['reference_hash']==json_hash(ref),'Frozen cohort reference differs')
        for seed in range(2001,2021):
            split=root/f'split_{seed}'
            require(sha(split/'lock.json')==lock['split_lock_hashes'][str(seed)],'Changed partition lock')
            member_record=json.loads((split/'membership.json').read_text())
            member=member_record['membership']
            split_lock=json.loads((split/'lock.json').read_text())
            require(split_lock['identity']==member_record['identity'] and not split_lock['test_scored_at_lock'],'Partition lock identity/timing differs')
            require(member_record['identity']==json_hash(dict(study=result['identity'],split_seed=seed,membership=member)),'Membership identity differs')
            fits=split_lock['fits']
            expected_fits={(s,a) for s in range(12,17) for a in expected_arms}
            require(len(fits)==len(expected_fits) and {(f['model_seed'],f['arm']) for f in fits}==expected_fits,'Missing or duplicated locked fits')
            for relative,expected in split_lock['files'].items():
                require(sha(split/relative)==expected,f'Changed fit/membership artifact: {relative}')
            hashes={role:json_hash(rows) for role,rows in member.items()}
            require(hashes==ref['revision_membership_hashes'][str(seed)],'Frozen cohort membership differs')
        key=(p['dataset'],p['model'])
        require(key not in seen,'Multiple complete runs of one task; explicitly choose one, do not pool')
        seen.add(key)
        frame=pd.read_csv(root/'metrics.csv',float_precision='round_trip')
        require(set(zip(frame.dataset,frame.model))=={key},'Metric task label mismatch')
        frames.append(frame);evidence.append(dict(dataset=key[0],model=key[1],protocol_sha256=sha(root/'protocol.json'),result_manifest_sha256=sha(path)))
    require(seen=={(d,m) for d in DATASETS for m in MODELS},'Missing complete tasks: '+str({(d,m) for d in DATASETS for m in MODELS}-seen))
    signatures=[json.loads((p.parent/'protocol.json').read_text())['source_signature']
                for p in Path(folder).rglob('RESULT_MANIFEST.json')
                if json.loads((p.parent/'protocol.json').read_text())['protocol'].get('experiment')==experiment]
    require(all(s==signatures[0] for s in signatures),'Implementation signatures differ between tasks')
    ratios={'Phish':553/1656,'Ponzi':269/804}
    table=analyze(pd.concat(frames,ignore_index=True),experiment,ratios,output)
    (Path(output)/'COLLECTION_CHECK.json').write_text(json.dumps(dict(status='PASS',experiment=experiment,tasks=evidence,comparisons=len(table),
        note='Compact export hashes are checked; omitted large native caches are not rehashed by this analysis.'),indent=2),encoding='utf-8')
    print(f'Collected ten complete tasks; Holm family size {len(table)}.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    old=sub.add_parser('replay-frozen');old.add_argument('--repo',type=Path,required=True);old.add_argument('--output',type=Path,required=True)
    new=sub.add_parser('collect');new.add_argument('--inputs',type=Path,required=True);new.add_argument('--experiment',choices=['direct','shuffle'],required=True);new.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='replay-frozen':replay_frozen(args.repo,args.output)
    else:collect_followup(args.inputs,args.experiment,args.output)
