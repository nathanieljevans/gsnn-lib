import omnipath as op 
import pandas as pd 
import copy 
import numpy as np


def get_interactions(undirected=False): 
    '''
    retrieve and process the omnipath interactions. 
    '''

    dorothea        = op.interactions.Dorothea().get()
    omnipath        = op.interactions.OmniPath().get()
    pathways_extra  = op.interactions.PathwayExtra().get()
    tf_mirna        = op.interactions.TFmiRNA().get()
    mirna           = op.interactions.miRNA().get()

    doro = dorothea.assign(source = lambda x: ['PROTEIN__' + y for y in x.source],
                        target = lambda x: ['RNA__' + y for y in x.target], 
                        edge_type = 'dorothea')[['source', 'target', 'edge_type']]

    omni = omnipath.assign(source = lambda x: ['PROTEIN__' + y for y in x.source],
                        target = lambda x: ['PROTEIN__' + y for y in x.target], 
                        edge_type = 'omnipath')[['source', 'target', 'edge_type']]

    path = pathways_extra.assign(source = lambda x: ['PROTEIN__' + y for y in x.source],
                        target = lambda x: ['PROTEIN__' + y for y in x.target], 
                        edge_type = 'pathways_extra')[['source', 'target', 'edge_type']]     

    tfmirna = tf_mirna.assign(source = lambda x: ['PROTEIN__' + y for y in x.source],
                        target = lambda x: ['RNA__' + y for y in x.target], 
                        edge_type = 'tf_mirna')[['source', 'target', 'edge_type']]

    mirna_ = mirna.assign(source = lambda x: ['RNA__' + y for y in x.source],
                        target = lambda x: ['RNA__' + y for y in x.target], 
                        edge_type = 'mirna')[['source', 'target', 'edge_type']] 
    
    _fdf = pd.concat((doro, omni, path, tfmirna, mirna_), axis=0, ignore_index=True)
    _fnames = _fdf.source.values.tolist() + _fdf.target.values.tolist()
    rna_space = [x.split('__')[1] for x in _fnames if x.split('__')[0] == 'RNA']
    protein_space = [x.split('__')[1] for x in _fnames if x.split('__')[0] == 'PROTEIN']
    RNA_PROT_OVERLAP = list(set(rna_space).intersection(set(protein_space)))
    trans = pd.DataFrame({'source': ['RNA__' + x for x in RNA_PROT_OVERLAP],
                        'target': ['PROTEIN__' + x for x in RNA_PROT_OVERLAP],
                        'edge_type':'translation'})
    print('# of translation (RNA->PROTEIN) edges:', len(trans))
    
    func_df = pd.concat((doro, omni, path, tfmirna, mirna_, trans), axis=0, ignore_index=True)
    
    if undirected: 
        print('making function graph undirected (adding reverse edges)')
        func_df2 = copy.deepcopy(func_df)
        func_df2 = func_df2.rename({'target':'source', 'source':'taret'}, axis=1)
        func_df = pd.concat((func_df, func_df2), ignore_index=True, axis=0)

    func_names = np.unique(func_df.source.tolist() + func_df.target.tolist()).tolist()

    return func_names, func_df
