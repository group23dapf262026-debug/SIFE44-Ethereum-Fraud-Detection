"""Fixed native GAT/SAGE backbones generalized only at the explicit feature input."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv,SAGEConv
from .utilities import require
from .features import ARMS,PURE36_INDICES,select_inputs
from .models_meta import HMSL

ARM_WIDTHS={'X14':14,'Pure36':36,'X14_Pure36':50,'X14_SIFE44':58}


def arm_columns(arm):
    """Map named arms into source14+SIFE44 moments; Pure36 never selects source14."""
    require(arm in ARM_WIDTHS,'Unknown input arm')
    if arm=='X14':return tuple(range(14))
    extra=tuple(14+i for i in (PURE36_INDICES if arm in ('Pure36','X14_Pure36') else range(44)))
    return extra if arm=='Pure36' else tuple(range(14))+extra


def transform_inputs(values,arm,model_name):
    """Signed-log only source14 for GAT; Pure36 and all extra columns keep their defined scale."""
    require(values.ndim==2 and values.shape[1]==ARM_WIDTHS[arm],'Input width mismatch')
    if model_name!='gat' or arm=='Pure36':return values
    source=values[:,:14]
    source=source.sign()*source.abs().log1p()
    return source if values.shape[1]==14 else torch.cat([source,values[:,14:]],dim=1)


class StudyGAT(nn.Module):
    """Native4-head GAT -> ELU ->4-head GAT2; frozen TRAIN normalization, no dropout/encoder."""
    def __init__(self,arm,hidden,statistics):
        super().__init__();self.arm=arm;self.input_width=ARM_WIDTHS[arm]
        columns=list(arm_columns(arm))
        self.register_buffer('input_mean',torch.tensor(statistics['mean'],dtype=torch.float32)[columns])
        self.register_buffer('input_scale',torch.tensor(statistics['scale'],dtype=torch.float32)[columns])
        require(bool(torch.isfinite(self.input_mean).all()) and bool((self.input_scale>0).all()),'Invalid GAT statistics')
        self.conv1=GATConv(self.input_width,hidden,heads=4,concat=True,dropout=0.,negative_slope=.2,add_self_loops=True,bias=True)
        self.conv2=GATConv(4*hidden,2,heads=4,concat=False,dropout=0.,negative_slope=.2,add_self_loops=True,bias=True)
    def forward(self,x,edge_index):
        h=(transform_inputs(x,self.arm,'gat')-self.input_mean)/self.input_scale
        return self.conv2(F.elu(self.conv1(h,edge_index)),edge_index)


class StudySAGE(nn.Module):
    """Raw source14/standardized extras -> Linear/tanh -> projected meanSAGE -> ReLU -> SAGE2."""
    def __init__(self,arm,hidden,statistics):
        super().__init__();self.arm=arm;self.input_width=ARM_WIDTHS[arm]
        columns=list(arm_columns(arm));mean=torch.tensor(statistics['mean'],dtype=torch.float32)[columns]
        scale=torch.tensor(statistics['scale'],dtype=torch.float32)[columns]
        # Preserve the reviewed raw-X14 backbone in every arm that has source14.
        if arm!='Pure36':mean[:14]=0.;scale[:14]=1.
        self.register_buffer('input_mean',mean);self.register_buffer('input_scale',scale)
        self.input_encoder=nn.Linear(self.input_width,hidden)
        self.conv1=SAGEConv(hidden,hidden,aggr='mean',project=True,normalize=False,root_weight=True,bias=True)
        self.conv2=SAGEConv(hidden,2,aggr='mean',project=True,normalize=False,root_weight=True,bias=True)
    def forward(self,x,edge_index):
        values=(x-self.input_mean)/self.input_scale
        h=torch.tanh(self.input_encoder(values))
        return self.conv2(F.relu(self.conv1(h,edge_index)),edge_index)


def build_classifier(model_name,arm,config,graph,target,statistics):
    if model_name=='gat':return StudyGAT(arm,config['hidden'],statistics)
    if model_name=='sage':return StudySAGE(arm,config['hidden'],statistics)
    require(model_name=='metaifd' and arm!='Pure36','Meta-IFD Pure36-only is unsupported: decoder needs source14')
    return HMSL(config['hidden'],2,graph,config['views'],target,input_width=ARM_WIDTHS[arm])


def lookup_features(feature_dir,node_type,ids):
    """Lookup exact full-incident SIFE44 for the sampled global node IDs."""
    keys=np.load(Path(feature_dir)/(node_type+'_ids.npy'),mmap_mode='r',allow_pickle=False)
    values=np.load(Path(feature_dir)/(node_type+'_sife44.npy'),mmap_mode='r',allow_pickle=False)
    ids=np.asarray(ids,dtype=np.int64);positions=np.searchsorted(keys,ids)
    require(bool((positions<len(keys)).all()) and np.array_equal(keys[positions],ids),'Missing node feature lookup')
    return np.asarray(values[positions],dtype=np.float64)


def forward_classifier(model,study,split,arm,batch,role):
    """Use the same sampled topology across all arms; Pure36 does not read source14."""
    p=study['protocol'];graph=study['data']['graph'];device=study['device'];model_name=p['model']
    ids=batch['node_ids'];directory=split['folder']/'cache'/role
    extra_raw={nt:lookup_features(directory/'features',nt,values.numpy()) for nt,values in ids.items()}
    edges={et:ix.to(device) for et,ix in batch['edge_index'].items()}
    if model_name=='metaifd':
        views={nt:torch.from_numpy(np.array(np.load(directory/'views'/f"{batch['batch_name']}_{nt}_views.npy",mmap_mode='r'),copy=True)).to(device)
               for nt in ('EOA','CA')}
        extra=None
        if arm!='X14':
            selected=list(PURE36_INDICES) if arm=='X14_Pure36' else list(range(44))
            mean=np.asarray(split['statistics']['mean'])[14:][selected]
            scale=np.asarray(split['statistics']['scale'])[14:][selected]
            extra={nt:torch.from_numpy(((value[:,selected]-mean)/scale).astype(np.float32)).to(device) for nt,value in extra_raw.items()}
        logits,contrast=model(views,edges,extra)
        return logits[:len(batch['seed_ids'])],contrast
    values=[]
    for nt in ('EOA','CA'):
        source=None if arm=='Pure36' else graph[nt].x[ids[nt]].numpy()
        value=select_inputs(source,extra_raw[nt],arm,model_name)
        values.append(torch.from_numpy(value.astype(np.float32)).to(device))
    offset={'EOA':0,'CA':len(ids['EOA'])}
    homogeneous=[ix+torch.tensor([[offset[et[0]]],[offset[et[2]]]],device=device) for et,ix in edges.items()]
    logits=model(torch.cat(values),torch.cat(homogeneous,dim=1));begin=offset[p['target']]
    return logits[begin:begin+len(batch['seed_ids'])],logits.new_zeros(())
