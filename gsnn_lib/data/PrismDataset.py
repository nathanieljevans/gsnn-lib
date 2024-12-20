
from torch.utils.data import Dataset

import numpy as np
import h5py
import torch
import time
import copy 



class PrismDataset(Dataset):
    def __init__(self, prism, prism_ids, data, clamp=False):
        '''
        '''
        super().__init__()

        self.prism      = prism.set_index('prism_id')
        self.prism_ids  = np.array([x for x in prism_ids if x not in ['nan']])
        self.data       = data
        self.clamp      = clamp

    def __len__(self):
        return len(self.prism_ids)

    def __getitem__(self, idx):

        prism_id      = self.prism_ids[idx]

        info        = self.prism.loc[prism_id]
        pert_id     = info.pert_id
        conc_um     = info.conc_um
        cell_iname  = info.cell_iname
        x_drug      = self.data.x_dict['drug_dict'][pert_id](conc_um)
        x_cell      = self.data.x_dict['cell_dict'][cell_iname]
        x           = x_drug + x_cell 

        y           = torch.tensor([info.cell_viab], dtype=torch.float32).view(1,1)
        
        if self.clamp: y = torch.clamp(y, min=0, max=1)

        return x.to_dense().detach(), y, prism_id
