"""Freeze the fixed20partition study and explicit strict memberships before fitting."""
import hashlib
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from .features import ARMS,FEATURE_VERSION,FEATURE_GROUPS,ARM_REMOVED_GROUP,selected_sife_indices
from .models import ARM_WIDTHS
from .tabular_inputs import LR_PROFILES
from .utilities import require,sha256,json_hash,read_json,write_json,runtime_information,verify_files

MODELS=('metaifd','gat','sage','lightgbm','rf')
TABULAR=('lightgbm','rf')


def fixed_configuration(model):
    """Predeclared configurations; no grid and no per-arm or per-dataset tuning."""
    return dict(metaifd=dict(hidden=128,lr=.001,views=3,contrast_weight=.1),
        gat=dict(hidden=32,lr=.01),sage=dict(hidden=32,lr=.01),
        lightgbm=dict(n_estimators=300,lr=.01,num_leaves=31),
        rf=dict(n_estimators=100,max_depth=10))[model].copy()


def supported_arms(model):
    require(model in MODELS,'Only the five agreed models are supported')
    return ARMS


def notebook_source_signature():
    shell=get_ipython();require(shell is not None,'Use a normal IPython/Jupyter notebook kernel')
    latest={}
    for source in shell.history_manager.input_hist_raw:
        first=source.splitlines()[0] if source.splitlines() else ''
        prefix='# SIFE20_SOURCE_CELL: '
        if first.startswith(prefix):latest[first[len(prefix):]]=hashlib.sha256(source.encode()).hexdigest()
    require(set(DEFINITION_KEYS)<=set(latest),'Run every definition cell above first')
    return {key:latest[key] for key in DEFINITION_KEYS}


def stratified_parts(roots,seed):
    """Address-disjoint60/20/20 using exactly the same split rule as the reviewed LR study."""
    positions=np.arange(len(roots['ids']));labels=roots['labels']
    development,test=train_test_split(positions,test_size=.2,random_state=seed,stratify=labels)
    train,validation=train_test_split(development,test_size=.25,random_state=seed,stratify=labels[development])
    parts=dict(train=train,validation=validation,test=test)
    combined=np.concatenate(list(parts.values()))
    require(len(combined)==len(set(combined))==len(labels),'Split coverage or overlap error')
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        require(not set(roots['addresses'][parts[a]])&set(roots['addresses'][parts[b]]),'Address overlap across roles')
    require(all(set(labels[pos])=={0,1} for pos in parts.values()),'Every role must include both classes')
    return parts


def open_study(data,output,dataset,model,device,source_signature,decoder_reference=None,
               run_mode='full',cpu_threads=2,lr_profile='signed_log1p'):
    """Freeze all scientific choices, data, seeds, versions and implementation identity."""
    require(model in MODELS and dataset in ('Phish','Ponzi'),'Unknown model/dataset')
    require(run_mode in ('full','smoke'),'Full or smoke mode required')
    require(lr_profile in LR_PROFILES,'Unknown fixed LR preprocessing profile')
    require(len(set(data['roots']['addresses']))==len(data['roots']['ids']),'Strict cohort contains duplicates')
    torch.set_num_threads(cpu_threads);neural=model not in TABULAR
    require(neural or str(device)=='cpu','Tabular models run on CPU')
    protocol=dict(study='sife20_group_ablation_five_fixed_v1',dataset=dataset,model=model,run_mode=run_mode,
        target='EOA' if dataset=='Phish' else 'CA',feature_version=FEATURE_VERSION,
        arms={arm:ARM_WIDTHS[arm] for arm in supported_arms(model)},
        unsupported_arms={},control_arm='X14_SIFE44',
        ablation_definition='remove named SIFE input columns; retain X14,typed topology,all other columns',
        feature_groups={name:list(indices) for name,indices in FEATURE_GROUPS.items()},
        kept_sife_columns={arm:list(selected_sife_indices(arm)) for arm in ARMS},
        removed_group=ARM_REMOVED_GROUP,
        pairing='newly fitted full control within this run; same sampled draws across all5arms; no prior-run cache/checkpoint import',
        split_seeds=list(range(2001,2021)) if run_mode=='full' else [2001],
        model_seeds=list(range(12,17)) if run_mode=='full' else [12],
        split_rule='strict unique-address cohort; stratified60/20/20; no released-mask score pooling',
        config=fixed_configuration(model),configuration_origin='fixed accepted-study budget; not older split-specific validation winners',
        hyperparameter_search=False,primary_metric='Macro-F1',decision='class1 probability>0.5; exacttieclass0; no threshold tuning',
        backbone={'gat':'native4head GAT ELU GAT2; no encoder/BN/dropout',
                  'sage':'Linear/tanh; projected mean SAGEConv h->h->2; no BN/dropout',
                  'metaifd':'public HMSL with optional extras appended after frozen14dim decoder'}.get(model,'standard tabular classifier'),
        lr_source14_profile=lr_profile if model=='lr' else None,
        preprocessing={'lr':('source14 signed-log1p; extras unchanged; TRAIN-root StandardScaler' if lr_profile=='signed_log1p' else
                            'rawsource14 reference; extras unchanged; TRAIN-root StandardScaler'),
            'rf':'rawsource14 and feature-defined extras; no scaler',
            'lightgbm':'rawsource14 and feature-defined extras; no scaler',
            'gat':'signed-log only source14; all active columns z-score from unique typed TRAIN sampled nodes; Pure36 has no source14',
            'sage':'rawsource14 unchanged; extras z-score from unique typed TRAIN sampled nodes before shared Linear/tanh',
            'metaifd':'raw14view generation; extras TRAIN-zscore concatenated algebraically after generation'}[model],
        max_epochs=(1000 if run_mode=='full' else 2) if neural else None,patience=50 if neural else None,
        checkpoint_rule='latest validation Macro-F1 tie; stop stale>50' if neural else 'one fit at fixed budget',
        weight_decay=.0005 if neural else None,batch_size=512 if run_mode=='full' else 8,
        fanouts=[100,100] if run_mode=='full' else [2,2],
        sampler_seed_rule='1201/1202/2301 +10000*(split_seed-2000); actual PyG directional2hop, shared across arms/seeds',
        feature_scope='exact full-incident typed graph records for roots and sampled neighbors; no labels/source14 in SIFE construction',
        test_policy='lock all arms and model seeds on every requested partition before any test sampling or scoring',
        seed_averaging='mean model seeds within split, then mean/sampleSD across20split means; do not treat100runs as independent',
        setting='static transductive; held-out node topology/features may be neighbors, labels not used as model input',
        limitations='exploratory reused corpus, no temporal/external transfer validation; Pure36 is scale-invariant, not money-information-free',
        parameter_budget='input width changes parameter count; report per-arm counts, not capacity matched',
        author_commit='439c6daa936f67f19486be76642e690a2153d0c5',
        meta_loss='public unsquaredL2 TripletMarginLoss mean margin.3; differs from paper equation',
        meta_runner='validation-best loading correction, explicit CPU view RNG and sampling seeds; not exact paper-table recovery',
        tabular_schema='named source14/SIFE DataFrames for fit and predict; exact column-order verification',
        development_provenance='group ablation designed after completed main-study results; all5models/both datasets retained; exploratory post-hoc on reused corpus,not newly blinded')
    identity=dict(protocol=protocol,data_manifest=data['manifest'],source_signature=source_signature,
        runtime=runtime_information(device),decoder_reference_hash=json_hash(decoder_reference) if decoder_reference else None)
    output=Path(output);output.mkdir(parents=True,exist_ok=True);path=output/'protocol.json'
    if path.exists():require(read_json(path)==identity,'Inputs/code/config/runtime changed; use a new output directory')
    else:write_json(path,identity)
    return dict(data=data,protocol=protocol,identity=json_hash(identity),output=output,
                device=torch.device(device),decoder_reference=decoder_reference)


def open_split(study,split_seed):
    """Persist exact IDs/addresses/labels for each role; same membership for every arm/model."""
    require(split_seed in study['protocol']['split_seeds'],'Unknown split seed')
    roots=study['data']['roots'];parts=stratified_parts(roots,split_seed)
    folder=study['output']/f'split_{split_seed}';folder.mkdir(exist_ok=True)
    membership={role:pd.DataFrame(dict(node_id=roots['ids'][pos],address=roots['addresses'][pos],
                label=roots['labels'][pos])) for role,pos in parts.items()}
    identity=json_hash(dict(study=study['identity'],split_seed=split_seed,
        membership={r:f.to_dict('records') for r,f in membership.items()}))
    record=dict(identity=identity,split_seed=split_seed,membership={r:f.to_dict('records') for r,f in membership.items()},
        sizes={r:len(f) for r,f in membership.items()},positives={r:int(f.label.sum()) for r,f in membership.items()})
    path=folder/'membership.json'
    if path.exists():require(read_json(path)==record,'Membership changed')
    else:
        write_json(path,record)
        for role,frame in membership.items():frame.to_csv(folder/(role+'_membership.csv'),index=False)
    return dict(folder=folder,seed=split_seed,parts=parts,membership=membership,identity=identity)


def verify_split_lock(study,split):
    """Check coverage and hashes for every arm/seed before evaluation or completed-fit reuse."""
    lock=read_json(split['folder']/'lock.json')
    require(lock['identity']==split['identity'],'Split lock identity changed')
    expected=set(itertools.product(study['protocol']['model_seeds'],study['protocol']['arms']))
    require({(row['model_seed'],row['arm']) for row in lock['fits']}==expected and len(lock['fits'])==len(expected),
            'Missing/duplicated arm or seed')
    verify_files(split['folder'],lock['files'])
    for role in ('train','validation'):
        root=split['folder']/'cache'/role;verify_files(root,read_json(root/'manifest.json')['files'])
    return lock


def verify_study_lock(study):
    path=study['output']/'study_lock.json';require(path.exists(),'Fit and freeze every requested partition before TEST')
    lock=read_json(path);require(lock['identity']==study['identity'],'Whole-study identity changed')
    require(set(lock['split_lock_hashes'])=={str(seed) for seed in study['protocol']['split_seeds']},'Missing partition lock')
    for seed in study['protocol']['split_seeds']:
        split=open_split(study,seed)
        require(sha256(split['folder']/'lock.json')==lock['split_lock_hashes'][str(seed)],'Partition lock changed')
        verify_split_lock(study,split)
    return lock
