"""Post-hoc descriptive stability and frozen, score-blind activity matching.

Run with --input-root pointing at the consolidated research folder. This reads
JSON/CSV/NPZ only: no checkpoint loading, training or changes to main results.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import platform
import zipfile
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import t
from sklearn.metrics import f1_score,precision_score,recall_score,average_precision_score,roc_auc_score

MODELS=['metaifd','gat','sage','lightgbm','rf']
METRICS=['macro_f1','binary_f1','precision','recall','fpr','average_precision','roc_auc']
LABELS=dict(metaifd='MetaIFD',gat='GAT',sage='GraphSAGE',lightgbm='LightGBM',rf='RF')

def digest(value):return hashlib.sha256(value).hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def require(value,message):
    if not bool(value):raise ValueError(message)

def optimal_pairs(positive,negative,caliper):
    """Maximum-cardinality matching, then minimum-distance, within caliper."""
    if len(positive)==0 or len(negative)==0:return np.empty((0,2),int),np.empty(0,float)
    distances=cdist(positive,negative)
    penalty=(len(positive)+1)*(caliper+1)
    costs=np.full((len(positive),len(negative)+len(positive)),penalty)
    costs[:,:len(negative)]=np.where(distances<=caliper,distances,2*penalty)
    rows,columns=linear_sum_assignment(costs)
    keep=(columns<len(negative));rows=rows[keep];columns=columns[keep]
    require(np.all(distances[rows,columns]<=caliper),'Ineligible matched pair')
    return np.column_stack((rows,columns)),distances[rows,columns]

def holm(pvalues):
    p=np.asarray(pvalues,dtype=float);order=np.argsort(p,kind='stable')
    corrected=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    output=np.empty_like(p);output[order]=corrected
    return output

def metrics(frame):
    y=frame.label.to_numpy();p=frame.prediction.to_numpy();s=frame.score.to_numpy()
    tn=int(np.sum((y==0)&(p==0)));fp=int(np.sum((y==0)&(p==1)))
    return dict(macro_f1=float(f1_score(y,p,average='macro',labels=[0,1],zero_division=0)),
                binary_f1=float(f1_score(y,p,zero_division=0)),precision=float(precision_score(y,p,zero_division=0)),
                recall=float(recall_score(y,p,zero_division=0)),fpr=fp/(fp+tn),
                average_precision=float(average_precision_score(y,s)),roc_auc=float(roc_auc_score(y,s)),
                n=len(frame),positives=int(y.sum()))

def summarize(frame):
    seedmeans=frame.groupby(['dataset','model','arm','split_seed'])[METRICS].mean().reset_index()
    rows=[]
    for (ds,model,arm),group in seedmeans.groupby(['dataset','model','arm']):
        for metric in METRICS:
            x=group[metric].dropna().to_numpy()*100
            if not len(x):continue
            rows.append(dict(dataset=ds,model=model,arm=arm,metric=metric,mean_percent=x.mean(),
                             sd_across_split_means_pp=x.std(ddof=1) if len(x)>1 else np.nan,
                             split_count=len(x),model_seeds_per_split=5))
    return seedmeans,pd.DataFrame(rows)

def differences(splits):
    rows=[]
    for (ds,model),group in splits.groupby(['dataset','model']):
        for metric in METRICS:
            pivot=group.pivot(index='split_seed',columns='arm',values=metric)
            for arm in pivot.columns:
                if arm=='X14':continue
                d=(pivot[arm]-pivot.X14).dropna().to_numpy()*100
                if not len(d):continue
                direction=-1 if metric=='fpr' else 1
                rows.append(dict(dataset=ds,model=model,arm=arm,metric=metric,splits=len(d),
                                 mean_delta_pp=d.mean(),sample_sd_delta_pp=d.std(ddof=1),median_delta_pp=np.median(d),
                                 empirical_p10_pp=np.quantile(d,.1),empirical_p90_pp=np.quantile(d,.9),
                                 min_delta_pp=d.min(),max_delta_pp=d.max(),
                                 wins=int(np.sum(direction*d>1e-10)),ties=int(np.sum(abs(d)<=1e-10)),losses=int(np.sum(direction*d<-1e-10))))
    return pd.DataFrame(rows)

def corrected_sensitivity(splits,sizes):
    rows=[]
    for (ds,model),group in splits.groupby(['dataset','model']):
        pivot=group.pivot(index='split_seed',columns='arm',values='macro_f1')
        for arm in ['X14_Pure36','X14_SIFE44']:
            d=((pivot[arm]-pivot.X14)*100).to_numpy();n=len(d)
            ratio=sizes[ds]['test']/sizes[ds]['train'];se=np.sqrt((1/n+ratio)*np.var(d,ddof=1))
            stat=float(d.mean()/se) if se else (float('inf') if d.mean() else 0.)
            p=float(2*t.sf(abs(stat),n-1))
            margin=t.ppf(.975,n-1)*se
            rows.append(dict(dataset=ds,model=model,arm=arm,mean_delta_pp=d.mean(),correction_ratio_test_train=ratio,
                             approximate_corrected_se_pp=se,pointwise95_low_pp=d.mean()-margin,pointwise95_high_pp=d.mean()+margin,
                             two_sided_p=p,df=n-1,scope='assumption-dependent sensitivity; graph dependence/nonadditive F1 not solved'))
    result=pd.DataFrame(rows);require(len(result)==20,'Expected20comparisons')
    result['holm_p_20_comparisons']=holm(result.two_sided_p)
    return result

def locate_inputs(root):
    found={}
    for path in sorted((root/'02_SIFE20_MAIN/02_RESULTS').rglob('*.zip')):
        with zipfile.ZipFile(path) as z:
            member=next(name for name in z.namelist() if name.endswith('/results/full/protocol.json') or name=='results/full/protocol.json')
            prefix=member[:-len('protocol.json')];protocol=json.loads(z.read(member))
            p=protocol['protocol'];key=p['dataset'],p['model']
            require(key not in found,'Duplicate model/dataset')
            manifest=json.loads(z.read(prefix+'RESULT_MANIFEST.json'))
            require(manifest['status']=='COMPLETE' and manifest['full_study'],'Incomplete study')
            require(p['study']=='sife20_native_five_fixed_v1' and p['run_mode']=='full','Wrong study')
            require(p['split_seeds']==list(range(2001,2021)) and p['model_seeds']==list(range(12,17)),'Wrong seed coverage')
            require(digest(z.read(member))==manifest['files']['protocol.json'],'Protocol hash')
            found[key]=dict(path=path,prefix=prefix,protocol=protocol,manifest=manifest)
    require(set(found)=={(ds,m) for ds in ('Phish','Ponzi') for m in MODELS},'Need exactly10runs')
    return found

def freeze_matching(inputs,output,plan):
    """This phase never opens predictions.csv, metrics.csv or per-model outcomes."""
    pairs=[];balances=[];scales=[];subsets={};expected={};sizes={}
    for ds in ('Phish','Ponzi'):
        info=inputs[ds,'lightgbm']
        with zipfile.ZipFile(info['path']) as z:
            rootfile='data/roots.npz' if info['prefix']=='results/full/' else info['prefix'].split('/results/')[0]+'/data/roots.npz';payload=z.read(rootfile)
            require(digest(payload)==info['protocol']['data_manifest']['roots_sha256'],'Roots hash')
            with np.load(io.BytesIO(payload),allow_pickle=False) as a:roots={k:a[k] for k in ('ids','labels','addresses','activity')}
            byid={int(node):i for i,node in enumerate(roots['ids'])}
            for seed in range(2001,2021):
                member=json.loads(z.read(info['prefix']+f'split_{seed}/membership.json'))['membership']
                expected[ds,seed]=member;sizes[ds]={r:len(v) for r,v in member.items()}
                train=np.array([byid[v['node_id']] for v in member['train']])
                test=np.array(sorted([byid[v['node_id']] for v in member['test']],key=lambda i:int(roots['ids'][i])))
                mu=roots['activity'][train].mean(axis=0);sd=roots['activity'][train].std(axis=0,ddof=0);sd=np.where(sd==0,1.,sd)
                standardized=(roots['activity']-mu)/sd
                positive=test[roots['labels'][test]==1];negative=test[roots['labels'][test]==0]
                matches,dist=optimal_pairs(standardized[positive],standardized[negative],plan['activity_matching']['caliper'])
                pos=positive[matches[:,0]] if len(matches) else np.array([],int)
                neg=negative[matches[:,1]] if len(matches) else np.array([],int)
                matched=np.concatenate((pos,neg));ids=roots['ids'][matched].astype(int)
                subsets[ds,seed]=set(ids.tolist())
                require(len(ids)==len(set(ids)),'Match replacement')
                require(not set(ids)&set(roots['ids'][train]),'TRAIN/TEST overlap')
                for pair_no,(i,j,d) in enumerate(zip(pos,neg,dist)):
                    pairs.append(dict(dataset=ds,split_seed=seed,pair_number=pair_no,positive_id=int(roots['ids'][i]),negative_id=int(roots['ids'][j]),
                                      positive_address=str(roots['addresses'][i]),negative_address=str(roots['addresses'][j]),distance=float(d),
                                      positive_log_count=roots['activity'][i,0],negative_log_count=roots['activity'][j,0],
                                      positive_log_value=roots['activity'][i,1],negative_log_value=roots['activity'][j,1]))
                balance=dict(dataset=ds,split_seed=seed,test_positive=len(positive),test_negative=len(negative),matched_pairs=len(pos),
                             positive_retention=len(pos)/len(positive),matched_roots=len(ids),small_under10_pairs=len(pos)<10,
                             mean_pair_distance=float(dist.mean()) if len(dist) else np.nan,max_pair_distance=float(dist.max()) if len(dist) else np.nan)
                for k,name in enumerate(('count','value')):
                    balance['before_standardized_gap_'+name]=float(standardized[positive,k].mean()-standardized[negative,k].mean())
                    balance['after_standardized_gap_'+name]=float(standardized[pos,k].mean()-standardized[neg,k].mean()) if len(pos) else np.nan
                    scales.append(dict(dataset=ds,split_seed=seed,variable=name,train_n=len(train),train_mean=mu[k],train_population_sd=sd[k]))
                balances.append(balance)
    frame=pd.DataFrame(pairs);frame.to_csv(output/'MATCHED_PAIRS.csv',index=False)
    pd.DataFrame(balances).to_csv(output/'MATCH_COVERAGE_BALANCE.csv',index=False)
    pd.DataFrame(scales).to_csv(output/'TRAIN_ACTIVITY_SCALING.csv',index=False)
    files=['ANALYSIS_PLAN.json','MATCHED_PAIRS.csv','MATCH_COVERAGE_BALANCE.csv','TRAIN_ACTIVITY_SCALING.csv']
    lock=dict(status='ALL40MATCH_MEMBERSHIPS_FROZEN_BEFORE_PREDICTION_READ',files={f:digest((output/f).read_bytes()) for f in files},
              members={f'{ds}/{seed}':sorted(ids) for (ds,seed),ids in subsets.items()})
    write_json(output/'MATCH_LOCK.json',lock)
    print('FROZEN matching for40partitions; pairs',len(pairs),flush=True)
    return subsets,expected,sizes

def evaluate(inputs,subsets,expected,output):
    matched=[];full=[];counts=[];source=[]
    lockhash=digest((output/'MATCH_LOCK.json').read_bytes())
    for (ds,model),info in inputs.items():
        p=info['protocol']['protocol'];prefix=info['prefix'];manifest=info['manifest']
        with zipfile.ZipFile(info['path']) as z:
            b=z.read(prefix+'metrics.csv');require(digest(b)==manifest['files']['metrics.csv'],'Metrics hash')
            existing=pd.read_csv(io.BytesIO(b));full.append(existing[existing.role=='test'])
            for seed in range(2001,2021):
                member=json.loads(z.read(prefix+f'split_{seed}/membership.json'))['membership']
                require(member==expected[ds,seed],'Models must share all split memberships')
                ids=subsets[ds,seed]
                expected_test=pd.DataFrame(member['test']).set_index('node_id')
                for model_seed in p['model_seeds']:
                    for arm in p['arms']:
                        path=f'split_{seed}/seed_{model_seed}/{arm}/predictions.csv';payload=z.read(prefix+path)
                        require(digest(payload)==manifest['files'][path],'Prediction hash')
                        frame=pd.read_csv(io.BytesIO(payload),float_precision='round_trip')
                        test=frame[frame.role=='test']
                        require(test.node_id.is_unique and set(test.node_id)==set(expected_test.index),'Test coverage')
                        require(np.array_equal(test.label.to_numpy(),expected_test.loc[test.node_id,'label'].to_numpy()),'Test labels')
                        require(np.array_equal(test.address.to_numpy(),expected_test.loc[test.node_id,'address'].to_numpy()),'Test addresses')
                        for key,value in dict(dataset=ds,model=model,split_seed=seed,model_seed=model_seed,arm=arm).items():
                            require(test[key].eq(value).all(),'Prediction identity '+key)
                        require(np.array_equal(test.prediction.to_numpy(),(test.score.to_numpy()>.5).astype(int)),'Decision changed')
                        subset=test[test.node_id.isin(ids)]
                        require(set(subset.node_id)==ids,'Matched subset incomplete')
                        if len(subset):
                            require(subset.label.sum()*2==len(subset),'Matched prevalence must be50percent')
                            matched.append(dict(dataset=ds,model=model,arm=arm,split_seed=seed,model_seed=model_seed,**metrics(subset)))
                        else:
                            matched.append(dict(dataset=ds,model=model,arm=arm,split_seed=seed,model_seed=model_seed,
                                                **{metric:np.nan for metric in METRICS},n=0,positives=0))
            counts.append(dict(dataset=ds,model=model,verified_frozen_prediction_files=20*5*len(p['arms'])))
            source.append(dict(dataset=ds,model=model,path=str(info['path']),protocol_sha256=manifest['files']['protocol.json'],
                               result_manifest_sha256=digest(z.read(prefix+'RESULT_MANIFEST.json'))))
        require(digest((output/'MATCH_LOCK.json').read_bytes())==lockhash,'Matching lock changed')
        print('SCORED frozen subsets',ds,model,counts[-1]['verified_frozen_prediction_files'],flush=True)
    full=pd.concat(full,ignore_index=True);matched=pd.DataFrame(matched)
    matched.to_csv(output/'MATCHED_PER_FIT_METRICS.csv',index=False)
    pd.DataFrame(counts).to_csv(output/'PREDICTION_COVERAGE.csv',index=False)
    write_json(output/'INPUT_PROVENANCE.json',source)
    return full,matched

def plot_results(output,full_splits,matched_splits,balance):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=False)
    for ax,ds in zip(axes,('Phish','Ponzi')):
        for j,model in enumerate(MODELS):
            x=full_splits[(full_splits.dataset==ds)&(full_splits.model==model)].pivot(index='split_seed',columns='arm',values='macro_f1')
            delta=(x.X14_SIFE44-x.X14)*100
            ax.scatter(j+np.linspace(-.18,.18,len(delta)),delta,s=20,alpha=.6,color='#3376b8')
            ax.scatter(j,delta.mean(),marker='D',color='black',s=42,zorder=4)
        ax.axhline(0,color='gray',linestyle='--',linewidth=1);ax.set_xticks(range(5),[LABELS[m] for m in MODELS],rotation=20)
        ax.set_title(ds);ax.set_ylabel('Macro-F1 difference (percentage points)')
    fig.suptitle('X14 + SIFE44 minus X14: 20 overlapping partitions; diamonds = means')
    fig.tight_layout();fig.savefig(output/'STABILITY_SPLIT_DELTAS.pdf');fig.savefig(output/'STABILITY_SPLIT_DELTAS.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,ds in zip(axes,('Phish','Ponzi')):
        for i,arm in enumerate(('X14','Pure36','X14_Pure36','X14_SIFE44')):
            means=[];errors=[]
            for model in MODELS:
                g=matched_splits[(matched_splits.dataset==ds)&(matched_splits.model==model)&(matched_splits.arm==arm)]
                means.append(g.macro_f1.mean()*100 if len(g) else np.nan);errors.append(g.macro_f1.std(ddof=1)*100 if len(g)>1 else np.nan)
            ax.errorbar(np.arange(5)+(i-1.5)*.12,means,yerr=errors,fmt='o',markersize=4,capsize=2,label=arm)
        ax.set_title(ds);ax.set_xticks(range(5),[LABELS[m] for m in MODELS],rotation=20);ax.set_ylabel('Matched TEST Macro-F1 (%)')
        ax.legend(fontsize=8)
    fig.suptitle('Activity-matched subsets: mean +/- SD across partition means (not CI)')
    fig.tight_layout();fig.savefig(output/'MATCHED_MACRO_F1.pdf');fig.savefig(output/'MATCHED_MACRO_F1.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    for ax,ds in zip(axes,('Phish','Ponzi')):
        b=balance[balance.dataset==ds]
        for i,name in enumerate(('count','value')):
            ax.scatter(np.arange(len(b))+.08*i,b['before_standardized_gap_'+name].abs(),marker='x',label=f'{name}: full test')
            ax.scatter(np.arange(len(b))+.08*i,b['after_standardized_gap_'+name].abs(),marker='o',s=18,label=f'{name}: matched')
        ax.axhline(.1,color='gray',linestyle='--');ax.set_title(ds);ax.set_xlabel('Partition index (seeds 2001-2020)');ax.set_ylabel('|Mean gap| / TRAIN SD');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'MATCH_BALANCE.pdf');fig.savefig(output/'MATCH_BALANCE.png',dpi=180);plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    planfile=Path(__file__).with_name('ANALYSIS_PLAN.json');plan=json.loads(planfile.read_text(encoding='utf-8'))
    (output/'ANALYSIS_PLAN.json').write_bytes(planfile.read_bytes())
    write_json(output/'EXECUTION_START.json',dict(plan_sha256=digest(planfile.read_bytes()),code_sha256=digest(Path(__file__).read_bytes()),
                                               python=platform.python_version(),started_before_any_subset_scoring=True))
    inputs=locate_inputs(args.input_root)
    subsets,expected,sizes=freeze_matching(inputs,output,plan)
    full,matched=evaluate(inputs,subsets,expected,output)
    require(len(full)==3800,'Full test coverage');require(len(matched)==3800,'Matched evaluation record coverage')
    fullsplits,fullsummary=summarize(full);matchsplits,matchsummary=summarize(matched)
    fullsplits.to_csv(output/'FULL_SPLIT_MEANS.csv',index=False);fullsummary.to_csv(output/'FULL_SUMMARY.csv',index=False)
    matchsplits.to_csv(output/'MATCHED_SPLIT_MEANS.csv',index=False);matchsummary.to_csv(output/'MATCHED_SUMMARY.csv',index=False)
    differences(fullsplits).to_csv(output/'FULL_PAIRED_STABILITY.csv',index=False)
    differences(matchsplits).to_csv(output/'MATCHED_PAIRED_STABILITY.csv',index=False)
    corrected_sensitivity(fullsplits,sizes).to_csv(output/'ASSUMPTION_DEPENDENT_INFERENCE_SENSITIVITY.csv',index=False)
    balance=pd.read_csv(output/'MATCH_COVERAGE_BALANCE.csv')
    plot_results(output,fullsplits,matchsplits,balance)
    files={p.relative_to(output).as_posix():digest(p.read_bytes()) for p in output.iterdir() if p.is_file()}
    write_json(output/'ANALYSIS_MANIFEST.json',dict(status='COMPLETE',full_fits=3800,matched_fit_evaluations=len(matched),
              training_performed=False,main_results_modified=False,matching_rules_changed_after_outcomes=False,files=files))
    print('COMPLETE',output,flush=True)

if __name__=='__main__':main()
