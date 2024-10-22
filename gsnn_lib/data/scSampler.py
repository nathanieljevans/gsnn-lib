
import os
import pandas as pd 
import torch 
import numpy as np

class scSampler: 
    '''Sample Px, Py from unperturbed and pertrubed single cell omics'''

    def __init__(self, root, drug_filter=None, cell_filter=None, test_prop=0., val_prop=0.):
        # TODO: how to handle partition splits... 

        self.root = root 
        data = torch.load(root + '/data.pt')
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
        self.test_prop = test_prop; self.val_prop = val_prop
        if test_prop > 0:
            self.meta = self.meta.assign(test = np.random.choice([True, False], size=self.meta.shape[0], p=[test_prop, 1-test_prop]))
        else: 
            self.meta = self.meta.assign(test = False)
        
        if val_prop > 0:    
            val_labels = np.zeros(self.meta.shape[0], dtype=bool)
            val_labels[self.meta[lambda x: ~x.test].sample(frac=val_prop).index] = True
            self.meta = self.meta.assign(val=val_labels)
        else:
            self.meta = self.meta.assign(val=False)
        
    def __len__(self): 
        return self.conditions.shape[0]

    def sample(self, idx, batch_size=64, partition='train'): 
        # TODO: Test that this is selecting appropriately 
        condition = self.conditions.iloc[idx]
        cell_line = condition.cell_line 
        pert_id = condition.drug
        dose = condition.dose / 1e6 # dose is in molars (conversion expects uM)
        self.controls.loc[cell_line]
        perts = self.meta.iloc[condition.id]
        ctrls = self.meta.iloc[self.controls.loc[cell_line].id]

        if partition == 'train': 
            perts = perts[lambda x: ~x.test]
            ctrls = ctrls[lambda x: ~x.test]
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
            
            perts = perts.sample(n=batch_size, replace=False, axis=0)

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

        # load outputs 
        y = [] 
        for fpath in perts.fpath: 
            y.append(torch.load(fpath))
        y = torch.stack(y, dim=0)

        return x,y 
        
