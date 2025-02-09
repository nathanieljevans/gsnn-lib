import numpy as np
import pandas as pd
import torch 


def get_geneid2uniprot(args): 
    uni2id = pd.read_csv(f'{args.extdata}/omnipath_uniprot2geneid.tsv', sep='\t').rename({'From':'uniprot', 'To':'gene_id'}, axis=1)
    uni2symb = pd.read_csv(f'{args.extdata}/omnipath_uniprot2genesymb.tsv', sep='\t').rename({'From':'uniprot', 'To':'gene_symbol'}, axis=1)
    gene_map = uni2id.merge(uni2symb, on='uniprot', how='outer')
    geneinfo = pd.read_csv(f'{args.data}/geneinfo_beta.txt', sep='\t')[lambda x: x.feature_space.isin(args.feature_space)].merge(gene_map, on=['gene_id'], how='left')
    geneinfo = geneinfo[['gene_id', 'uniprot']].drop_duplicates().dropna()
    if args.lincs is not None: 
        lincs2uniprot = geneinfo[lambda x: x.uniprot.isin(args.lincs)].set_index('gene_id')['uniprot'].drop_duplicates().to_dict()
    else: 
        lincs2uniprot = geneinfo.set_index('gene_id')['uniprot'].drop_duplicates().to_dict()
    
    return lincs2uniprot


def get_function_pathway_features(data, uni2rea, K=100): 

    _t = uni2rea[['pathway_id', 'name']].drop_duplicates()
    id2name = {k:v for k,v in zip(_t['pathway_id'], _t['name'])}
    uni2rea = uni2rea[lambda x: x.species == 'Homo sapiens'][['uniprot', 'pathway_id']]

    func_uni = []
    for f in data['node_names_dict']['function']: 
        if 'COMPLEX' in f: 
            for x in f.split(':')[1].split('_'): 
                func_uni.append(x)
        else: 
            func_uni.append( f.split('__')[1])
    func_uni = np.unique(func_uni) 
    
    uni2rea = uni2rea[lambda x: x.uniprot.isin(func_uni)]

    # select the top K most commonly annotated pathways 
    pathways = uni2rea.groupby('pathway_id').count().sort_values('uniprot', ascending = False).index[:K].tolist()
    uni2rea = uni2rea[lambda x: x.pathway_id.isin(pathways)].drop_duplicates() 

    x = []

    for f in data['node_names_dict']['function']: 
        xx = torch.zeros(len(pathways))
        if 'COMPLEX' in f: 
            uni = f.split(':')[1].split('_')
        else: 
            uni = [f.split('__')[1]]

        for i,row in uni2rea[lambda x: x.uniprot.isin(uni)].iterrows(): 

            idx = pathways.index(row.pathway_id)
            xx[idx] = 1. 

        x.append(xx)

    x = torch.stack(x, dim=0)

    pathnames = [id2name[id] for id in pathways]

    return x, pathways, pathnames 


def get_x_drug_conc(uM, idx, N, eps): 

    x_drug_conc = torch.zeros((N), dtype=torch.float32)
    x_val = (np.log10(uM + eps) - np.log10(eps))/(-np.log10(eps))
    x_drug_conc[idx] = x_val

    return x_drug_conc.to_sparse()

def get_dep2iname(path):
    clue_info = pd.read_csv(f'{path}/cellinfo_beta.txt', sep='\t')[['cell_iname', 'ccle_name']].dropna()
    ccle_info = pd.read_csv(f'{path}/ccle_info.txt', sep=',')[['DepMap_ID', 'CCLE_Name']].rename({'CCLE_Name':'ccle_name'}, axis=1).dropna()
    dep2iname = clue_info.merge(ccle_info)[['DepMap_ID', 'cell_iname']].drop_duplicates()
    return dep2iname

def get_ccle2iname(path):
    return pd.read_csv(f'{path}/cellinfo_beta.txt', sep='\t')[['cell_iname', 'ccle_name']].dropna()


def load_cellinfo(root, cellspace=None): 
    '''
    combine ccle and lincs cell line info
    input 
        cell_line_space (list): `cell_iname` identifiers of cell lines to return
    output: 
        (dataframe): merged dataframe 
        (dict): dep2iname 
        (dict): iname2dep
    '''
    cinfo = pd.read_csv(f'{root}/ccle_info.txt', low_memory=False)
    linfo = pd.read_csv(f'{root}/cellinfo_beta.txt', sep='\t', low_memory=False)

    cellinfo = cinfo.merge(linfo, left_on='CCLE_Name', right_on='ccle_name', how='inner')
    cellinfo = cellinfo[['DepMap_ID', 'cell_iname', 'sex', 'cell_lineage', 'primary_disease_x', 'subtype', 'growth_pattern', 'doubling_time', 'donor_age', 'primary_or_metastasis']]
    
    if cellspace is not None: cellinfo = cellinfo[lambda x: x.cell_iname.isin(cellspace)]

    dep2iname = cellinfo.set_index('DepMap_ID')['cell_iname'].to_dict() 
    iname2dep = cellinfo.set_index('cell_iname')['DepMap_ID'].to_dict()

    return cellinfo, dep2iname, iname2dep

def filter_to_common_cellspace(omics, cell_space=None): 
    '''
    takes a list of omics dataframes and filters to the common subset of cell_inames. Optionally filter to overlap with `cell_space` if not None. 
    '''
    if cell_space is None: 
        lines = None
    else: 
        lines = set(list(cell_space))

    for omic in omics: 
        if lines == None: 
            lines = set(omic.index.tolist())
        else: 
            lines = lines.intersection(set(omic.index.tolist()))

    cell_space = np.sort(list(lines))

    return [omic[lambda x: x.index.isin(cell_space)] for omic in omics], cell_space

def aggregate_duplicate_genes(omic): 
    '''
    non-unique mapping from gene symbol to uniprot id can result in multiple gene identifiers (columns)
    '''

def impute_missing_gene_ids(omic, gene_space, fill_value): 
    '''
    impute the missing gene identifiers with the given `fill_value`, then select `gene_space`
    '''

    omic_genes = omic.columns.tolist() 
    missing_genes = list( set(list(gene_space)) - set(omic_genes) )
    
    if len(missing_genes) > 0: 
        print('\t# missing genes:', len(missing_genes), f'[impute value: {fill_value}]')
        missing_df = pd.DataFrame({**{'cell_iname':omic.index}, **{g:[fill_value]*omic.shape[0] for g in missing_genes}}).set_index('cell_iname')
        omic = omic.merge(missing_df, left_index=True, right_index=True, how='inner', validate='1:1')

    return omic[list(gene_space)]




def split_ids_by_attribute(ids, attributes, proportions):
    '''
    1. group `ids` by `attribute` 
    2. order by size of attribute (# ids in attribute)
    3. randomly select an id from each attribute group for each split until the right proportions are reached. 
    
    This ensures that the largest attributes will be equally represented in each split but also ensures for id splits. 
    NOTE: If there are more attribute groups than a given split size, then the smaller attirbute groups will not be represented. 

    Args: 
        ids             listlike            ids to split; len N 
        attributes      listlike            respective id attributes; len N 
        proportions     listlike            number of splits and proportions to assign to each split; 
                                            the first split group proportion may not perfectly match due to integer rounding/casting errors.
                                            len S

    Returns: 
        splits          list of lists       groups of ids assigned to each split; len S 
    ''' 
    assert np.isclose(sum(proportions), 1), 'proportions do not sum to 1'
    attributes = np.array(attributes)
    ids = list(ids)
    _ids = np.sort(ids)
    attrs, cnts = np.unique(attributes, return_counts=True)
    attrs = attrs[np.argsort(cnts)]      # ensure that the attribute groups are sorted by number of ids in each attribute. 
    grouped_ids = [np.array(ids)[attributes == a].tolist() for a in attrs]   # cell lines grouped by attribute in sorted order; same order as attr 

    n = len(ids)
    split_sizes = [np.round(p*n) for p in proportions[1:]]
    split_sizes = [n - sum(split_sizes)] + split_sizes 
    split_sizes = [int(x) for x in split_sizes]

    splits = [[] for i in proportions]

    while len(ids) > 0: 
        for group in grouped_ids: 
            for split, size in zip(splits, split_sizes): 
                if (len(split) < size) & (len(ids) > 0) & (len(group) > 0): 
                    np.random.shuffle(group)
                    id = group.pop()
                    split.append(id) 
                    ids.remove(id)
                    
    assert sum([len(x) for x in splits]) == n, 'union of splits are a different size than original id list'

    test = [] 
    for s in splits: 
        test+=s 
    assert (np.sort(test) == _ids).all(), 'splits are not unique or are missing ids'

    return splits