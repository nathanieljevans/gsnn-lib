
import pandas as pd
from gsnn_lib.proc.lincs.utils import load_cellinfo
import numpy as np
import torch

def load_prism(root, cellspace=None, drugspace=None, avg_replicates=True): 

    cellinfo, dep2iname, iname2dep = load_cellinfo(root=root)

    prism_primary = pd.read_csv(f'{root}/primary-screen-replicate-collapsed-logfold-change.csv')
    prism_primary = prism_primary.rename({'Unnamed: 0':'depmap_id'}, axis=1)

    prism_primary = prism_primary.assign(cell_iname = lambda x: [dep2iname[xx] if xx in dep2iname else None for xx in x.depmap_id])

    prism_primary = prism_primary.set_index(['depmap_id', 'cell_iname']).stack().reset_index().rename({'level_2':'meta', 0:'log_fold_change'}, axis=1)
    prism_primary[['pert_id_long', 'pert_dose','_']] = prism_primary.meta.str.split(pat='::', n=2, expand=True)
    prism_primary = prism_primary.assign(pert_id = lambda x: [xx[:13] for xx in x.pert_id_long])

    prism_primary = prism_primary.assign(screen_id = 'primary')

    prism_secondary = pd.read_csv(f'{root}/secondary-screen-replicate-collapsed-logfold-change.csv')
    prism_secondary = prism_secondary.rename({'Unnamed: 0':'depmap_id'}, axis=1)

    prism_secondary = prism_secondary.assign(cell_iname = lambda x: [dep2iname[xx] if xx in dep2iname else None for xx in x.depmap_id])

    prism_secondary = prism_secondary.set_index(['depmap_id', 'cell_iname']).stack().reset_index().rename({'level_2':'meta', 0:'log_fold_change'}, axis=1)
    prism_secondary[['pert_id_long', 'pert_dose','_']] = prism_secondary.meta.str.split(pat='::', n=2, expand=True)
    prism_secondary = prism_secondary.assign(pert_id = lambda x: [xx[:13] for xx in x.pert_id_long])

    prism_secondary = prism_secondary.assign(screen_id = 'secondary')

    prism = pd.concat((prism_primary, prism_secondary), axis=0)
    prism = prism[lambda x: ~x.cell_iname.isna()]

    # filter to drugspace + cellspace 
    if cellspace is not None: prism = prism[lambda x: x.cell_iname.isin(cellspace)]
    if drugspace is not None: prism = prism[lambda x: x.pert_id.isin(drugspace)]

    if avg_replicates: 
        prism = prism.groupby(['pert_id', 'depmap_id', 'cell_iname', 'pert_dose']).agg({'log_fold_change':np.mean, 'screen_id':list}).reset_index()
        prism = prism.assign(num_repl = lambda x: [len(xx) for xx in x.screen_id])

    # create aggregate id 
    prism['prism_id'] = prism[['cell_iname', 'pert_id', 'pert_dose']].agg('::'.join, axis=1)

    # add cell viability transformation
    # Calculate viability data as two to the power of replicate-level logfold change data
    prism = prism.assign(cell_viab = lambda x: 2**(x.log_fold_change))

    prism.pert_dose = prism.pert_dose.astype(float)

    prism = prism.rename({'pert_dose':'conc_um'}, axis=1)
    prism = prism[['prism_id', 'pert_id', 'conc_um', 'cell_iname', 'cell_viab', 'log_fold_change']]

    return prism