import torch 
import numpy as np 
#from src.models.SparseLinear import SparseLinear
from src.models.SparseLinear2 import SparseLinear2 as SparseLinear
from src.models import utils
from src.models.GroupLayerNorm import GroupLayerNorm
from src.models.GSNN import * 
import torch_geometric as pyg

'''
State mediated GSNN: 
General idea, let's infer molecular entity state independant of cell signaling. 
Therefore, we infer a molecular entity vector that acts as an attention mask and mediates distinct signaling behavior during the GSNN forward pass. 
As it current stands, we couldn't really use the GSNN model for a neural ODE, but with an approach like smGSNN we could. 

The more I think about it... GNN approach is probably not a great approach for parameterizing the function to infer node state since
we have a homogenous network wrt to omic nodes 
If we want to use a GNN for this approach, should probably use a heterogenous network - right now can't distinguish MUT from EXPR, etc... 

A better approach might be to use a small 2-3 layer GSNN... altho we would need multioutput node prediction which the GSNN isn't set up well for 

ugh... I guess a simple AE approach might work as well. no longer assumes any topology constraints, but maybe that's okay...
 topology would still be maintained in cell signaling 
'''
# ugh this is harder then I thought ... 
# for prototyping we could just set omic nodes to zero (instead of removing the nodes entirely...)
# so we would subset to get omic_edge_index, which would be used to infer node state mediating vector 
# then for each obs, set all omic nodes to zero for GSNN forward pass 
# could simplify further and just set the pert nodes to zero instead of making new edge index ... 
class smGNN(torch.nn.Module): 
    def __init__(self, edge_index, node_names, channels, function_nodes, fix_hidden_channels, dropout): 
        super().__init__()

        if not fix_hidden_channels: raise NotImplementedError

        self.N = len(node_names)
        self.E = edge_index.size(1)
        self.channels = channels 
        self.register_buffer('function_nodes', function_nodes)
        self.register_buffer('edge_index', edge_index )
        self.register_buffer('drug_node_mask', torch.tensor(np.array(['DRUG__' in x for x in node_names]), dtype=torch.bool))

        self.gnn = pyg.nn.models.GAT(in_channels=1, 
                                        hidden_channels=32, 
                                        num_layers=2,
                                        out_channels=channels, 
                                        dropout=dropout, 
                                        act='elu',
                                        act_first=False,
                                        act_kwargs=None,
                                        norm='layer',              # ??? batch 
                                        norm_kwargs=None,
                                        jk='last')
        
        
    def forward(self, x): 

        # UGHHH we have to batch and unbatch x for pyg - could introduce large memory use ?
        B = x.size(0)
        x = x.clone()
        x[:, self.drug_node_mask, :] *= 0  # set all drug nodes to zero so they can't affect the state vector 
        x_batched = x.view(-1,1)
        edge_index_batched = self.edge_index.repeat(1, B).reshape(2, -1) + (torch.arange(B, device=x.device).repeat_interleave(self.E)*self.N).unsqueeze(0)
        out = self.gnn(x_batched, edge_index_batched)

        # predict mask
        out = torch.sigmoid(out)

        # unbatch x 
        out = out.view(B, -1, self.channels)

        # select function nodes only to match GSNN latent layers 
        # out shape: (B, N, C)
        out = out[:, self.function_nodes, :]

        # reshape to match GSNN function node latent state - TODO: check that this indexing is accurate! 
        # (I believe) the GSNN latent state should be in format: (f1_1, f1_2, ... f1_c, f2_1, ... f2_c, ...)
        # I'm guess that this indexing is current wrong - would help explain why our `r_cell` performance is so low ... 
        #out = out.view(B, -1, 1)
        out = out.permute(0, 2, 1).reshape(B, -1, 1)

        return out
    


class smAE(torch.nn.Module): 
    def __init__(self, node_names, channels, function_nodes, fix_hidden_channels, dropout, latent_dim=50):
        
        super().__init__()
        if not fix_hidden_channels: raise NotImplementedError

        #self.register_buffer('drug_node_mask', torch.tensor(np.array(['DRUG__' in x for x in node_names]), dtype=torch.bool))
        self.register_buffer('omic_node_mask', torch.tensor(np.array([x.split('__')[0] in ['EXPR', 'MUT', 'CNV', 'METHYL'] for x in node_names]), dtype=torch.bool))

        n_omics = self.omic_node_mask.sum()
        self.function_nodes = function_nodes
        self.channels = channels

        self.ae = torch.nn.Sequential(torch.nn.Linear(n_omics, latent_dim), 
                                      torch.nn.Dropout(dropout), 
                                      torch.nn.ELU(), 
                                      torch.nn.Linear(latent_dim, len(function_nodes)*channels))
        
        self.norm = torch.nn.InstanceNorm1d(len(function_nodes)*channels, affine=False)

    def forward(self, x): 

        x = x.clone()
        #x[:, self.drug_node_mask, :] *= 0  # set all drug nodes to zero so they can't affect the state vector 
        x = x[:, self.omic_node_mask, 0]

        B = x.size(0)

        out = self.ae(x).view(B, -1) #len(self.function_nodes), self.channels)

        #out = torch.nn.functional.softmax(out, dim=-1)

        #out = out.permute(0, 2, 1).reshape(B, -1, 1)

        out = torch.sigmoid(self.norm(out)).unsqueeze(-1)

        return out
    





class smResBlock(torch.nn.Module): 

    def __init__(self, edge_index, channels, function_nodes, fix_hidden_channels, bias, nonlin, residual, two_layers, dropout=0., norm='layer', init='xavier'): 
        super().__init__()
        assert norm in ['layer', 'none'], 'unrecognized `norm` type'
        
        w1_indices, w2_indices, w3_indices, w1_size, w2_size, w3_size, channel_groups = get_conv_indices(edge_index, channels, function_nodes, fix_hidden_channels)
        
        self.two_layers = two_layers
        self.dropout = dropout
        self.residual = residual

        self.lin1 = SparseLinear(indices=w1_indices, size=w1_size, bias=bias, init=init)
        if norm == 'layer': self.norm1 = GroupLayerNorm(channel_groups)
        if two_layers: 
            self.lin2 = SparseLinear(indices=w2_indices, size=w2_size, bias=bias)
            if norm == 'layer':self.norm2 = GroupLayerNorm(channel_groups)
        self.lin3 = SparseLinear(indices=w3_indices, size=w3_size, bias=bias, init=init)

        self.nonlin = nonlin()

    def forward(self, x, s): 

        out = self.lin1(x) 
        
        if hasattr(self, 'norm1'): out = self.norm1(out) 

        out = s*out # state-mediated behavior 
        out = self.nonlin(out) 

        # TODO: should this ^^ be a multihead attention block?? 
        out = torch.nn.functional.dropout(out, p=self.dropout, training=self.training)

        if self.two_layers: 
            out = self.lin2(out) 
            if hasattr(self, 'norm2'): out = self.norm2(out)
            out = out*s # state-mediated behavior 
            out = self.nonlin(out) 
            out = torch.nn.functional.dropout(out, p=self.dropout, training=self.training)

        out = self.lin3(out) 
        if self.residual: out = out + x 

        return out
    
        

class smGSNN(torch.nn.Module): 

    def __init__(self, edge_index, node_names, channels, input_node_mask, output_node_mask, layers, residual=True, dropout=0., 
                            nonlin=torch.nn.ELU, bias=True, share_layers=False, fix_hidden_channels=False, two_layer_conv=False, 
                                add_function_self_edges=False, norm='layer', init='xavier'):
        super().__init__()

        self.share_layers = share_layers            # whether to share function node parameters across layers
        self.register_buffer('output_node_mask', output_node_mask)
        self.input_node_mask = input_node_mask
        self.layers = layers 
        self.residual = residual
        self.channels = channels
        self.add_function_self_edges = add_function_self_edges
        self.fix_hidden_channels = fix_hidden_channels 
        self.two_layer_conv = two_layer_conv

        self.register_buffer('omic_node_mask', torch.tensor(np.array([x.split('__')[0] in ['EXPR', 'MUT', 'CNV', 'METHYL'] for x in node_names]), dtype=torch.bool))

        function_nodes = (~(input_node_mask | output_node_mask)).nonzero(as_tuple=True)[0]
        if add_function_self_edges: 
            print('Augmenting `edge index` with function node self-edges.')
            edge_index = torch.cat((edge_index, torch.stack((function_nodes, function_nodes), dim=0)), dim=1)
        self.register_buffer('edge_index', edge_index)
        self.E = self.edge_index.size(1)                             # number of edges 
        self.N = torch.unique(self.edge_index.view(-1)).size(0)      # number of nodes

        self.register_buffer('function_edge_mask', torch.isin(edge_index[0], function_nodes)) # edges from a function node / e.g., not an input or output edge 
        self.register_buffer('input_edge_mask', self.input_node_mask[self.edge_index[0]].type(torch.float32))

        self.dropout = dropout

        _n = 1 if self.share_layers else self.layers
        self.ResBlocks = torch.nn.ModuleList([smResBlock(self.edge_index, channels, function_nodes, fix_hidden_channels, 
                                                       bias, nonlin, residual=residual, two_layers=two_layer_conv, 
                                                       dropout=dropout, norm=norm, init=init) for i in range(_n)])
        
        #self.state_fn = smGNN(edge_index, node_names, channels, function_nodes, fix_hidden_channels, dropout=dropout)
        self.state_fn = smAE(node_names, channels, function_nodes, fix_hidden_channels, dropout, latent_dim=100)
        


    def forward(self, x, mask=None, return_last_activation=False, return_all_activations=False):
        '''
        Assumes x is `node` indexed 
        ''' 

        s = self.state_fn(x)
        #print(s[0, :, 0])
        #print(s.mean(), s.std()) # s is almost all 0/1 - clearly saturated and not learning. Maybe batch normalization would help? 

        # set omic nodes to zero 
        x[:, self.omic_node_mask, :] *= 0.

        x = utils.node2edge(x, self.edge_index)  # convert x to edge-indexed
        x0 = x
        if return_all_activations: activations = [x0]
        for l in range(self.layers): 
            x = self.ResBlocks[0 if self.share_layers else l](x, s)
            if not self.residual: x += x0
            if mask is not None: x = mask * x
            if return_all_activations: activations.append(x)

        if self.residual: x /= self.layers

        if return_last_activation: 
            return x
        elif return_all_activations: 
            return torch.cat(activations, dim=-1)
        else: 
            return utils.edge2node(x, self.edge_index, self.output_node_mask)  # convert x from edge-indexed to node-indexed

