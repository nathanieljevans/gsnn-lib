
from torch.utils.data import Dataset

import numpy as np
import h5py
import torch
import time
import copy 



class LincsDataset(Dataset):
    def __init__(self, root, cond_ids, data, cond_meta, input_shuffle_idxs=None, output_shuffle_idxs=None):
        '''

        '''
        super().__init__()

        self.cond_ids               = np.array(cond_ids)
        self.root                   = root 
        self.data                   = data
        self.input_shuffle_idxs     = input_shuffle_idxs
        self.output_shuffle_idxs    = output_shuffle_idxs
        
        cond_meta                   = cond_meta[lambda x: x.condition_id.isin(self.cond_ids)]
        cond_meta                   = cond_meta.set_index('condition_id')[['pert_id', 'pert_dose', 'cell_iname']]
        self.cond_meta              = cond_meta

    def __len__(self):
        return len(self.cond_ids)

    def __getitem__(self, idx):

        cond_id      = self.cond_ids[idx]

        info        = self.cond_meta.loc[cond_id]
        pert_id     = info.pert_id
        conc_um     = info.pert_dose
        cell_iname  = info.cell_iname
        x_drug      = self.data.x_dict['drug_dict'][pert_id](conc_um)
        x_cell      = self.data.x_dict['cell_dict'][cell_iname]
        x           = x_drug + x_cell 
        y           = torch.load(f'{self.root}/obs/{cond_id}.pt', weights_only=True)

        # convert to dense tensor
        x           = x.to_dense().detach() 
        y           = y.to_dense().detach()

        # input randomization
        if self.input_shuffle_idxs is not None:
            x = x[self.input_shuffle_idxs]

        # output randomization s
        if self.output_shuffle_idxs is not None:
            y = y[self.output_shuffle_idxs]

        return x, y, cond_id
