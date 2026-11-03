"""Collect completed result ZIPs without mixing full/smoke or different scientific definitions."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import pandas as pd


def read_result(archive_path,allow_smoke=False):
    mode='smoke' if allow_smoke else 'full'
    with zipfile.ZipFile(archive_path) as archive:
        roots=[name[:-len('protocol.json')] for name in archive.namelist()
               if name==f'results/{mode}/protocol.json' or name.endswith(f'/results/{mode}/protocol.json')]
        if len(roots)!=1:raise ValueError('Expected one result working tree or notebook export in '+str(archive_path))
        root=roots[0]
        protocol=json.loads(archive.read(root+'protocol.json'));p=protocol['protocol']
        if p['study']!='sife20_group_ablation_five_fixed_v1' or p['run_mode']!=mode:raise ValueError('Wrong study/mode: '+str(archive_path))
        result=json.loads(archive.read(root+'RESULT_MANIFEST.json'))
        if result['status']!='COMPLETE':raise ValueError('Partial result: '+str(archive_path))
        # Verify exported result tables and every per-fit evaluation/prediction artifact.
        for name,expected in result['files'].items():
            if hashlib.sha256(archive.read(root+name)).hexdigest()!=expected:raise ValueError('Changed artifact '+name)
        summary=pd.read_csv(io.BytesIO(archive.read(root+'summary.csv')))
        splits=pd.read_csv(io.BytesIO(archive.read(root+'split_means.csv')))
    return protocol,result,summary,splits


def main():
    parser=argparse.ArgumentParser();parser.add_argument('inputs',nargs='+',help='Completed Kaggle result ZIPs')
    parser.add_argument('--output',required=True);parser.add_argument('--allow-smoke',action='store_true')
    args=parser.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    identities={};sources=set();features=set();summaries=[];split_tables=[];runs=[]
    for path in map(Path,args.inputs):
        protocol,result,summary,splits=read_result(path,args.allow_smoke);p=protocol['protocol'];key=(p['dataset'],p['model'])
        if key in identities:
            if identities[key]==result['identity']:continue
            raise ValueError('More than one different run for '+str(key)+'; select the declared primary run explicitly')
        identities[key]=result['identity'];sources.add(json.dumps(protocol['source_signature'],sort_keys=True));features.add(p['feature_version'])
        summaries.append(summary);split_tables.append(splits)
        runs.append(dict(dataset=key[0],model=key[1],identity=result['identity'],source=str(path),runtime=protocol['runtime']))
    if len(sources)!=1 or len(features)!=1:raise ValueError('Do not pool different feature/scientific code versions')
    combined=pd.concat(summaries,ignore_index=True);split_frame=pd.concat(split_tables,ignore_index=True)
    combined.to_csv(output/'ALL_SUMMARY.csv',index=False);split_frame.to_csv(output/'ALL_SPLIT_MEANS.csv',index=False)
    macro=combined[combined.metric=='macro_f1'].copy()
    macro['mean_sd']=macro.apply(lambda row:f"{row.mean_percent:.3f} Â± {row.sd_across_split_means_pp:.3f}",axis=1)
    table=macro.pivot(index=['dataset','model'],columns='arm',values='mean_sd').reset_index()
    table.to_csv(output/'MACRO_F1_TABLE.csv',index=False)
    deltas=[]
    for (dataset,model),part in split_frame.groupby(['dataset','model']):
        pivot=part.pivot(index='split_seed',columns='arm',values='macro_f1')
        for arm in pivot.columns:
            if arm=='X14_SIFE44':continue
            change=(pivot[arm]-pivot['X14_SIFE44'])*100
            deltas.append(dict(dataset=dataset,model=model,arm=arm,mean_removed_minus_full_pp=float(change.mean()),mean_contribution_full_minus_removed_pp=float(-change.mean()),
                               sd_delta_across_splits_pp=float(change.std(ddof=1)),splits=len(change)))
    pd.DataFrame(deltas).to_csv(output/'PAIRED_GROUP_CONTRIBUTIONS.csv',index=False)
    (output/'COLLECTION_MANIFEST.json').write_text(json.dumps(dict(runs=runs,mode='smoke' if args.allow_smoke else 'full',
        expected_full_runs=10,collected_runs=len(runs),all_five_models_on_both_datasets=len(runs)==10,
        statistical_scope='descriptive split-level paired deltas; no uncorrected100run significance claim'),indent=2),encoding='utf-8')
    print(table.to_string(index=False));print('Saved comparison:',output)

if __name__=='__main__':main()
