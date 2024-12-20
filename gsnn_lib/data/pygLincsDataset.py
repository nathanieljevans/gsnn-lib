
from torch.utils.data import Dataset
import torch_geometric as pyg
import numpy as np
import h5py
import torch
import time
import copy 



class pygLincsDataset(Dataset):
    def __init__(self, root, cond_ids, data, condinfo):
        '''

        '''
        super().__init__()

        self.cond_ids  = np.array(cond_ids)
        self.root     = root 
        self.data     = data
        
        condinfo  = condinfo[lambda x: x.condition_id.isin(self.cond_ids)]
        condinfo  = condinfo.set_index('condition_id')[['pert_id', 'pert_dose', 'cell_iname']]
        self.condinfo = condinfo

    def __len__(self):
        return len(self.cond_ids)

    def __getitem__(self, idx):

        # create data object 
        cond_id      = self.cond_ids[idx]

        info        = self.condinfo.loc[cond_id]
        pert_id     = info.pert_id
        conc_um     = info.pert_dose
        cell_iname  = info.cell_iname
        x_drug      = self.data.x_dict['drug_dict'][pert_id](conc_um)
        x_cell      = self.data.x_dict['cell_dict'][cell_iname]
        x           = x_drug + x_cell 
        y           = torch.load(f'{self.root}/obs/{cond_id}.pt', weights_only=True)

        data = pyg.data.HeteroData()

        for key,edge_index in self.data.edge_index_dict.items():
            data[key].edge_index = edge_index
        
        # add self edges 
        data['input','to','input'].edge_index = torch.stack([torch.arange(len(self.data.node_names_dict['input'])), 
                                                             torch.arange(len(self.data.node_names_dict['input']))], dim=0)
        data['function','to','function'].edge_index = torch.stack([torch.arange(len(self.data.node_names_dict['function'])),
                                                                     torch.arange(len(self.data.node_names_dict['function']))], dim=0)
        data['output','to','output'].edge_index = torch.stack([torch.arange(len(self.data.node_names_dict['output'])), 
                                                             torch.arange(len(self.data.node_names_dict['output']))], dim=0)

        data['input'].x = x.to_dense().view(-1,1)
        data['function'].x = torch.zeros(len(self.data.node_names_dict['function']), 1)
        data['output'].x = torch.zeros(len(self.data.node_names_dict['output']), 1)

        data['input'].y = torch.zeros(len(self.data.node_names_dict['input']), 1)
        data['function'].y = torch.zeros(len(self.data.node_names_dict['function']), 1)
        data['output'].y = y.to_dense().view(-1,1)

        data.sig_id = cond_id

        return data
