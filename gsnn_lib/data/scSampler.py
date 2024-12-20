
import os
import pandas as pd 
import torch 
import numpy as np

class scSampler: 
    '''Sample Px, Py from unperturbed and pertrubed single cell omics'''

    def __init__(self, root, drug_filter=None, cell_filter=None):
        # TODO: how to handle partition splits... 

        self.root = root 
        data = torch.load(root + '/data.pt')
        self.data = data
        self.dose_f_dict = data['x_dict']['drug_dict'] # dict to function: dose_f_dict[drug][dose_value_um] -> x_drug_inputs

        obs = os.listdir(self.root + '/PROC/')
        meta = {'pert_type':[], 'id':[], 'adata_idx':[], 'dose':[], 'drug':[], 'cell_line':[], 'fpath':[]}
        for i,o in enumerate(obs): 
            typ, idx, pert_id, dose, line = o[:-2].split('_')
            meta['pert_type'].append(typ)
            meta['adata_idx'].append(idx)
            meta['dose'].append(float(dose))
            meta['drug'].append(pert_id)
            meta['cell_line'].append(line)
            meta['fpath'].append(self.root + '/PROC/' + o)
            meta['id'].append(i)
        self.meta = pd.DataFrame(meta)

        self.conditions = self.meta[lambda x: x.pert_type == 'perturbed'].groupby(['drug', 'dose', 'cell_line'])[['id']].agg(list).reset_index()
        if drug_filter is not None: self.conditions = self.conditions[lambda x: x.drug.isin(drug_filter)]
        if cell_filter is not None: self.conditions = self.conditions[lambda x: x.cell_line.isin(cell_filter)]
        
        self.controls = self.meta[lambda x: x.pert_type == 'unperturbed'].groupby(['cell_line'])[['id']].agg(list)

        # in-distribution test/val assignments 
        train_idx = torch.load(root + '/train_idxs.pt')
        test_idx = torch.load(root + '/test_idxs.pt')
        val_idx = torch.load(root + '/val_idxs.pt')

        self.meta = self.meta.assign(train = False, test = False, val = False)
        self.meta.loc[train_idx, 'train'] = True
        self.meta.loc[test_idx, 'test'] = True
        self.meta.loc[val_idx, 'val'] = True

        # check that train,test,val are disjoint
        assert self.meta[lambda x: x.train & x.test].shape[0] == 0, 'train and test overlap'
        assert self.meta[lambda x: x.train & x.val].shape[0] == 0, 'train and val overlap'
        assert self.meta[lambda x: x.test & x.val].shape[0] == 0, 'test and val overlap'
        
        self.drug_idxs = torch.tensor([i for i,n in enumerate(data['node_names_dict']['input']) if 'DRUG_' in n], dtype=torch.long)
        self.n_drugs = len(self.drug_idxs)
        self.n_cell_lines = len(self.conditions.cell_line.unique())
        self.cell2onehot = {cell:x for cell,x in zip(self.conditions.cell_line.unique(), torch.eye(self.n_cell_lines))}
    
    def sample_from_all_profiles(self, partition='train', verbose=True, N=None):
        '''return unperturbed and perturbed profiles'''
        
        if partition == 'train':
            meta = self.meta[lambda x: ~x.test & ~x.val]
        elif partition == 'test':
            meta = self.meta[lambda x: x.test]
        elif partition == 'val':
            meta = self.meta[lambda x: x.val]
        else:
            raise ValueError('partition must be "train" or "test" or "val"')
        
        if N is not None: 
            if meta.shape[0] < N:
                N = meta.shape[0]
            meta = meta.sample(n=N, replace=False, axis=0)
        
        x = [] ; ii=0
        for i,row in meta.iterrows():
            if verbose: print(f'loading data... {ii}/{meta.shape[0]}', end='\r'); ii+=1
            xx = torch.load(row.fpath)
            if row.pert_type == 'unperturbed':
                xx = xx[self.data.X2Y0_idxs]
            x.append(xx)

        return torch.stack(x, dim=0)

    def __len__(self): 
        return self.conditions.shape[0]
    
    def sample(self, batch_size, partition='train'):

        cond_idxs = torch.randint(0, len(self), size=(batch_size,))    
        x=[]; y=[]; x_cell=[]; x_drug=[]; y0=[]
        for idx in cond_idxs:
            xx ,yy, xx_cell, xx_drug, yy0 = self.sample_(idx.item(), batch_size=1, partition=partition)
            x.append(xx)
            y.append(yy)
            x_cell.append(xx_cell)
            x_drug.append(xx_drug)
            y0.append(yy0)
        return torch.cat(x, dim=0), torch.cat(y, dim=0), torch.cat(x_cell, dim=0), torch.cat(x_drug, dim=0), torch.cat(y0, dim=0)
    
    def sample_targets(self, idx, batch_size=64, partition='train'): 
        condition = self.conditions.iloc[idx]
        perts = self.meta.iloc[condition.id]

        if partition == 'train': 
            perts = perts[lambda x: (~x.test) & (~x.val)]
        elif partition == 'test':
            perts = perts[lambda x: x.test]
        elif partition == 'val':
            perts = perts[lambda x: x.val]
        else:
            raise ValueError('partition must be "train" or "test" or "val"')
        
        if perts.shape[0] > batch_size: 
            perts = perts.sample(n=batch_size, replace=False, axis=0)

        # load outputs 
        y = [] 
        for fpath in perts.fpath: 
            y.append(torch.load(fpath))
        y = torch.stack(y, dim=0)

        return y 
    
    def sample_controls(self, idx, batch_size=64, partition='train'):

        condition = self.conditions.iloc[idx]
        cell_line = condition.cell_line 
        ctrls = self.meta.iloc[self.controls.loc[cell_line].id]

        if partition == 'train': 
            ctrls = ctrls[lambda x: (~x.test) & (~x.val)]
        elif partition == 'test':
            ctrls = ctrls[lambda x: x.test]
        elif partition == 'val':
            ctrls = ctrls[lambda x: x.val]
        else:
            raise ValueError('partition must be "train" or "test" or "val"')
        
        if ctrls.shape[0] > batch_size:
            ctrls = ctrls.sample(n=batch_size, replace=False, axis=0)

        x = [] 
        for fpath in ctrls.fpath: 
            xx = torch.load(fpath)
            x.append(xx)
        x = torch.stack(x, dim=0)

        y0 = x[:, self.data.X2Y0_idxs] 

        return y0



    def sample_(self, idx, batch_size=64, partition='train', ret_all_y=False, ret_all_y0=False): 
        # TODO: Test that this is selecting appropriately 
        condition = self.conditions.iloc[idx]
        cell_line = condition.cell_line 
        pert_id = condition.drug
        dose = condition.dose / 1e3 # dose is in nanomolars (conversion expects uM)
        perts = self.meta.iloc[condition.id]
        ctrls = self.meta.iloc[self.controls.loc[cell_line].id]

        if partition == 'train': 
            perts = perts[lambda x: (~x.test) & (~x.val)]
            ctrls = ctrls[lambda x: (~x.test) & (~x.val)]
        elif partition == 'test':
            perts = perts[lambda x: x.test]
            ctrls = ctrls[lambda x: x.test]
        elif partition == 'val':
            perts = perts[lambda x: x.val]
            ctrls = ctrls[lambda x: x.val]
        else:
            raise ValueError('partition must be "train" or "test" or "val"')

        if batch_size is not None: # otherwise return all
            if perts.shape[0] < batch_size: 
                batch_size = perts.shape[0]
            
            if not ret_all_y: perts = perts.sample(n=batch_size, replace=False, axis=0)

            if not ret_all_y0: 
                if ctrls.shape[0] < batch_size:
                    ctrls = ctrls.sample(n=batch_size, replace=True, axis=0) # this is unlikely to occur
                else:
                    ctrls = ctrls.sample(n=batch_size, replace=False, axis=0)

        # load inputs 
        x = [] 
        for fpath in ctrls.fpath: 
            xx = torch.load(fpath)
            xx = xx + self.dose_f_dict[pert_id](dose)
            x.append(xx)
        x = torch.stack(x, dim=0)

        x_drug = x[:, self.drug_idxs]
        x_cell = self.cell2onehot[cell_line].unsqueeze(0).expand(x.size(0), -1)

        # load outputs 
        y = [] 
        for fpath in perts.fpath: 
            y.append(torch.load(fpath))
        y = torch.stack(y, dim=0)

        y0 = x[:, self.data.X2Y0_idxs]
        # X,y, x_cell, x_drug, y0
        return x ,y, x_cell, x_drug, y0
        


'''


'''