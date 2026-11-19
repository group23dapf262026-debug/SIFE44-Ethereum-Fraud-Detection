from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn
from .utilities import require,sha256,read_json,write_json,atomic_torch_save

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
