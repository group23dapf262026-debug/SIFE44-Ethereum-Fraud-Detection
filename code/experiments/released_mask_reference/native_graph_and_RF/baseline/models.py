"""X14 backbones restored from the group's audited native GAT and encoded mean SAGE."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv,SAGEConv
from .models_meta import HMSL
from .utilities import require,sha256,read_json,write_json,atomic_torch_save


class SignedLogGAT(nn.Module):
    """Audited GAT: signed-log/TRAIN z-score -> native4-head GAT -> ELU -> GAT2."""
    def __init__(self,hidden,statistics=None):
        super().__init__();self.input_width=14
        mean=torch.zeros(14) if statistics is None else torch.tensor(statistics['mean'],dtype=torch.float32)
        scale=torch.ones(14) if statistics is None else torch.tensor(statistics['scale'],dtype=torch.float32)
        require(mean.shape==scale.shape==(14,) and bool(torch.isfinite(mean).all()) and
                bool(torch.isfinite(scale).all() and (scale>0).all()),'Invalid frozen GAT normalization')
        self.register_buffer('input_mean',mean);self.register_buffer('input_scale',scale)
        self.conv1=GATConv(14,hidden,heads=4,concat=True,dropout=0.,negative_slope=.2,add_self_loops=True,bias=True)
        self.conv2=GATConv(4*hidden,2,heads=4,concat=False,dropout=0.,negative_slope=.2,add_self_loops=True,bias=True)
    def normalize_input(self,x):
        require(x.ndim==2 and x.shape[1]==14,'Expected unchanged source14')
        transformed=x.sign()*x.abs().log1p()
        return (transformed-self.input_mean)/self.input_scale
    def forward(self,x,edge_index):
        h=self.normalize_input(x)
        return self.conv2(F.elu(self.conv1(h,edge_index)),edge_index)


class InputEncodedSAGE(nn.Module):
    """Audited SAGE: rawX14 -> Linear/tanh -> mean SAGE(project=True) -> ReLU -> SAGE2."""
    def __init__(self,hidden):
        super().__init__();self.input_width=14
        self.input_encoder=nn.Linear(14,hidden)
        self.conv1=SAGEConv(hidden,hidden,aggr='mean',project=True,normalize=False,root_weight=True,bias=True)
        self.conv2=SAGEConv(hidden,2,aggr='mean',project=True,normalize=False,root_weight=True,bias=True)
    def encode_input(self,x):
        require(x.ndim==2 and x.shape[1]==14,'Expected unchanged source14')
        return torch.tanh(self.input_encoder(x))
    def forward(self,x,edge_index):
        h=self.encode_input(x)
        return self.conv2(F.relu(self.conv1(h,edge_index)),edge_index)


class FrozenInteractionDecoder(nn.Module):
    """Public pretrained conditional decoder: latent50 + source14 + relation6 ->14."""
    def __init__(self):
        super().__init__();self.latent_size=50
        self.MLP=nn.Sequential()
        self.MLP.add_module('L0',nn.Linear(70,32));self.MLP.add_module('A0',nn.Tanh())
        self.MLP.add_module('L_out0',nn.Linear(32,14))
    def forward(self,z,source,relation):
        return torch.sigmoid(self.MLP(torch.cat([z,source,relation],dim=-1)))


def decoder_tensor_fingerprint(state):
    digest=hashlib.sha256()
    for name in sorted(state):
        tensor=state[name].detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(tuple(tensor.shape)).encode());digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def materialize_decoder(output,reference):
    """Restore visible numeric weights and verify their independent tensor fingerprint."""
    state={name:torch.tensor(values,dtype=torch.float32) for name,values in reference['tensors'].items()}
    require(decoder_tensor_fingerprint(state)==reference['tensor_fingerprint'],'Inline public decoder tensors changed')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/'public_decoder.pt'
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=True)
        require(decoder_tensor_fingerprint(saved)==reference['tensor_fingerprint'],'Cached decoder changed')
    else:atomic_torch_save(state,path)
    decoder=FrozenInteractionDecoder();decoder.load_state_dict(state,strict=True)
    decoder.eval();decoder.requires_grad_(False)
    return decoder


def author_row_normalize(values):
    inverse=1/values.sum(dim=1);inverse[torch.isinf(inverse)]=0
    return values*inverse[:,None]


def save_meta_views(graph,records,output,decoder,role_seed,views):
    """Cache public-code conditional interaction views; RNG/traversal are declared corrections."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    generator=torch.Generator(device='cpu').manual_seed(role_seed+50000)
    files={}
    for record in records:
        batch=torch.load(record['path'],map_location='cpu',weights_only=True)
        name=Path(record['path']).stem
        for nt in ('EOA','CA'):
            source=graph[nt].x[batch['node_ids'][nt]].cpu()
            require(len(source)>0,'Meta-IFD requires both node types in each sampled batch')
            path=output/f'{name}_{nt}_views.npy'
            values=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(views,4,len(source),14))
            for view in range(views):
                interaction=0
                for rid,et in enumerate(graph.edge_types):
                    if et[0]!=nt:continue
                    for start in range(0,len(source),65536):
                        part=source[start:start+65536]
                        z=torch.randn((len(part),50),generator=generator)
                        condition=torch.zeros((len(part),6));condition[:,rid]=1
                        with torch.no_grad():generated=author_row_normalize(decoder(z,part,condition))
                        require(bool(torch.isfinite(generated).all()),'Nonfinite generated view')
                        values[view,interaction,start:start+len(part)]=generated.numpy()
                    interaction+=1
                require(interaction==3,'Expected three outgoing relation types')
                values[view,3]=source.numpy()
            values.flush();values._mmap.close();del values
            files[path.name]=sha256(path)
    return files


def build_model(graph,model_name,config,target,statistics=None):
    if model_name=='gat':return SignedLogGAT(config['hidden'],statistics)
    if model_name=='sage':return InputEncodedSAGE(config['hidden'])
    require(model_name=='metaifd','Unknown neural model')
    return HMSL(config['hidden'],2,graph,config['views'],target,input_width=14)


def forward_batch(model,model_name,graph,batch,view_dir,device):
    """Only X14 or generated source14 views enter the classifier."""
    ids=batch['node_ids'];edges={et:ix.to(device) for et,ix in batch['edge_index'].items()}
    if model_name=='metaifd':
        name=batch['batch_name']
        views={nt:torch.from_numpy(np.array(np.load(Path(view_dir)/f'{name}_{nt}_views.npy',mmap_mode='r'),copy=True)).to(device)
               for nt in ('EOA','CA')}
        logits,contrast=model(views,edges)
        return logits[:len(batch['seed_ids'])],contrast
    source={nt:graph[nt].x[ids[nt]].to(device) for nt in ('EOA','CA')}
    offset={'EOA':0,'CA':len(ids['EOA'])}
    homogeneous=[ix+torch.tensor([[offset[et[0]]],[offset[et[2]]]],device=device) for et,ix in edges.items()]
    logits=model(torch.cat([source['EOA'],source['CA']]),torch.cat(homogeneous,dim=1))
    begin=offset[batch['target_type']]
    return logits[begin:begin+len(batch['seed_ids'])],logits.new_zeros(())
