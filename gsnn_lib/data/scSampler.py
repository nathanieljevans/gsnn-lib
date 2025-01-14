
import os
import pandas as pd 
import torch 
import numpy as np

class scSampler: 
    '''Sample Px, Py from unperturbed and pertrubed single cell omics'''

    def __init__(self, root, pert_ids, ctrl_ids, batch_size, shuffle=True, ret_all_targets=True):
        # TODO: how to handle partition splits... 

        self.root = root 
        self.data = torch.load(root + '/data.pt')
        self.dose_f_dict = self.data['x_dict']['drug_dict'] # dict to function: dose_f_dict[drug][dose_value_um] -> x_drug_inputs
        self.batch_size = batch_size

        # load meta data and filter to partition (train, test, val)
        self.drugmeta = pd.read_csv(self.root + '/drug_meta.csv').loc[pert_ids]
        self.drugmeta = self.drugmeta.assign(fpath = [f'perturbed_{i}_{row.pert_id}_{row.dose_value}_{row.cell_line}.pt' for i,row in self.drugmeta.iterrows()])
        self.ctrlmeta = pd.read_csv(self.root + '/ctrl_meta.csv').loc[ctrl_ids]
        self.ctrlmeta = self.ctrlmeta.assign(fpath = [f'unperturbed_{i}_{row.pert_id}_{row.dose_value}_{row.cell_line}.pt' for i,row in self.ctrlmeta.iterrows()])

        # agg all indices by condition ; should produce a list of indices for each condition
        self.cond2drugids = self.drugmeta.groupby(['pert_id', 'dose_value', 'cell_line']).apply(lambda x: x.index.tolist()).reset_index().rename(columns={0:'id'})
        self.line2ctrlids = self.ctrlmeta.groupby(['cell_line']).apply(lambda x: x.index.tolist()).reset_index().rename(columns={0:'id'}).set_index('cell_line')

        self.drug_idxs = torch.tensor([i for i,n in enumerate(self.data['node_names_dict']['input']) if 'DRUG_' in n], dtype=torch.long)
        self.n_drugs = len(self.drug_idxs)

        self.ret_all_targets = ret_all_targets
        self.shuffle = shuffle

    def load_all(self): 
        '''load all data'''
        X = [] 
        Y = [] 
        Y0 = [] 
        for cond_idx in range(self.__len__()): 
            x, y, y0 = self.sample(cond_idx, ret_all_y=True, ret_all_y0=True)
            X.append(x)
            Y.append(y)
            Y0.append(y0)

        return torch.cat(X, dim=0), torch.cat(Y, dim=0), torch.cat(Y0, dim=0), self.cond2drugids[['cell_line', 'pert_id', 'dose_value']]
        
    def __len__(self): 
        return self.cond2drugids.shape[0]
    
    def __iter__(self):

        # TODO: fix ret_all to be a parameter 
        # TODO: add randomization 
        if self.shuffle: 
            conds = np.random.permutation(self.__len__())
        else: 
            conds = range(self.__len__())
            
        for cond_idx in conds: 
            yield self.sample(cond_idx, ret_all_y=self.ret_all_targets, ret_all_y0=False)

    def sample_targets(self, cond_idx, ret_all=False): 
        '''this loads the perturbed data for a given condition; will load a randomly sampled subset unless ret_all=True'''
        condition = self.cond2drugids.iloc[cond_idx]

        cell_line = condition.cell_line 
        pert_id = condition.pert_id
        dose_um = condition.dose_value / 1e3 # dose is in nanomolars (conversion expects uM)
        ids = condition.id

        perts = self.drugmeta.loc[ids]
        
        if not ret_all and (perts.shape[0] > self.batch_size): 
            perts = perts.sample(n=self.batch_size, replace=False, axis=0)

        # load outputs 
        y = [] 
        for fpath in perts.fpath: 
            y.append(torch.load(self.root + '/PROC/' + fpath))
        y = torch.stack(y, dim=0)

        return y, cell_line, pert_id, dose_um
    
    def sample_inputs(self, cond_idx, ret_all=False):
        '''this loads the unperturbed data, however, some inputs are "drug" features and will be zero'''

        condition = self.cond2drugids.iloc[cond_idx]
        cell_line = condition.cell_line 
        pert_id = condition.pert_id
        dose_um = condition.dose_value / 1e3 # dose is in nanomolars (conversion expects uM)

        ids = self.line2ctrlids.loc[cell_line].id

        ctrls = self.ctrlmeta.loc[ids]

        if not ret_all and (ctrls.shape[0] > self.batch_size):
            ctrls = ctrls.sample(n=self.batch_size, replace=False, axis=0)

        # load inputs
        x = []
        for fpath in ctrls.fpath:
            x.append(torch.load(self.root + '/PROC/' + fpath))
        x = torch.stack(x, dim=0)

        # add drug features
        x = x + self.dose_f_dict[pert_id](dose_um).to_dense().unsqueeze(0).expand(x.shape[0], -1)
        
        y0 = x[:, self.data.X2Y0_idxs].detach()

        return x, y0, cell_line, pert_id, dose_um

    def sample(self, cond_idx, ret_all_y=False, ret_all_y0=False): 
            
        y, cell_line1, pert_id1, dose_um1 = self.sample_targets(cond_idx, ret_all=ret_all_y)
        x, y0, cell_line2, pert_id2, dose_um2 = self.sample_inputs(cond_idx, ret_all=ret_all_y0)
        
        assert cell_line1 == cell_line2, 'cell line mismatch'
        assert pert_id1 == pert_id2, 'pert id mismatch'
        assert dose_um1 == dose_um2, 'dose value mismatch'
        
        # x = (y0 + drug_encoded)
        # y = perturbed expression 
        # y0 = unperturbed expression
        return x, y, y0
        


'''


'''