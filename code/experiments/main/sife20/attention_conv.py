"""Author interaction message sum, expressed with native PyTorch index_add.

Same trainable tensors and equations as reference_author/attention_conv.py.
The unused author skip parameter is retained for state-dict comparison.
No optional scatter/sparse extension is needed for this operation.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import Linear
from torch_geometric.nn.inits import reset


class my_conv(nn.Module):
    def __init__(self, in_channels, out_channels, metadata, group='sum'):
        super().__init__()
        if not isinstance(in_channels, dict):
            in_channels = {nt: in_channels for nt in metadata[0]}
        if group != 'sum':
            raise ValueError('The released model uses relation sum aggregation')
        self.out_channels = out_channels
        self.v_lin = nn.ModuleDict({nt: Linear(width, out_channels) for nt,width in in_channels.items()})
        self.a_lin = nn.ModuleDict({nt: Linear(width+out_channels, out_channels) for nt,width in in_channels.items()})
        self.skip = nn.ParameterDict({nt: nn.Parameter(torch.ones(1)) for nt in in_channels})

    def reset_parameters(self):
        reset(self.v_lin)
        reset(self.a_lin)
        for parameter in self.skip.values():
            nn.init.ones_(parameter)

    def forward(self, x_dict, edge_index_dict):
        projected = {nt: self.v_lin[nt](values) for nt,values in x_dict.items()}
        incoming = {nt: [] for nt in x_dict}
        for (src, _, dst), edges in edge_index_dict.items():
            summed = projected[dst].new_zeros((len(projected[dst]), self.out_channels))
            summed.index_add_(0, edges[1], projected[src][edges[0]])
            incoming[dst].append(summed)
        output = {}
        for nt, values in x_dict.items():
            total = torch.stack(incoming[nt], dim=0).sum(dim=0)
            output[nt] = self.a_lin[nt](torch.tanh(torch.cat([total, values], dim=1)))
        return output
