import os 
import urllib
import scperturb as scp 
import scanpy as sc
import pandas as pd
import numpy as np 
from scipy.stats import ttest_ind, norm


def load_scp_dataset(data='../../../data/', dataset='SrivatsanTrapnell2020_sciplex3'):
    '''
    Dataset options: 
        SrivatsanTrapnell2020_sciplex3
        AissaBenevolenskaya2021
        ...
    '''

    if not os.path.exists(f'{data}/{dataset}.h5ad'): 
        print(f'downloading {dataset}...')
        urllib.request.urlretrieve(f"https://zenodo.org/record/7041849/files/{dataset}.h5ad", f"{data}/{dataset}.h5ad")

    adata = scp.utils.sc.read_h5ad(f'{data}/{dataset}.h5ad')

    return adata

def get_AissaBenevolenskaya2021(genespace=None, data='../../../data/', max_prct_ribo=100, max_prct_mito=100, min_ngenes_per_cell=100, 
                     min_cells_per_gene=500, time=72, min_ncounts=100):
    '''
    
    ''' 

    dataset = 'AissaBenevolenskaya2021'

    adata = load_scp_dataset(data, dataset)

    # filter gene vars by cell count 
    adata = adata[:, adata.var.ncells > min_cells_per_gene]

    # filter cells by gene count 
    adata = adata[adata.obs.ngenes > min_ngenes_per_cell]

    # filter high ribo 
    adata = adata[adata.obs.percent_ribo <= max_prct_ribo]

    # filter high mito 
    adata = adata[adata.obs.percent_mito <= max_prct_mito]

    # filter cells by ncounts 
    adata = adata[adata.obs.ncounts >= min_ncounts]

    # ensure critical attribures are not nan 
    adata = adata[~adata.obs.time.isnull()]

    # filter to time point 
    if time is not None: 
        adata = adata[adata.obs.time == time]

    if genespace is not None: 
        adata = adata[:, adata.var.ensembl_id.isin(genespace)]


    return adata


def select_most_perturbed_genes(adata, N=1000, K=100, alpha=0.1):
    # assumes data has been zscore already 

    diffs = []

    ii = 0 ; ncond = adata.obs['cell_line'].nunique() * adata.obs['pert_id'].nunique()
    for cell_line in adata.obs['cell_line'].unique():

        for pert_id in adata.obs['pert_id'].dropna().unique(): 
            print(f'computing gene perturbation significance...[{cell_line}, {pert_id}] ~ {100*ii/ncond:.2f}%', end='\r')
            Y = adata[(adata.obs['cell_line'] == cell_line) & (adata.obs['pert_id'] == pert_id)].X.todense()
            Y0 = adata[(adata.obs['cell_line'] == cell_line) & (adata.obs['perturbation'] == 'control')].X.todense()
            Y = np.asarray(Y); Y0 = np.asarray(Y0)

            stat, pval = ttest_ind(Y, Y0, axis=0, equal_var=False)

            diffs.append(1.*(pval < alpha))
            ii+=1
    print()

    diffs = np.vstack(diffs) 
    diffs = diffs.mean(axis=0)
    rank_idxs = np.argsort(-diffs)

    return adata[:, rank_idxs[:N]]



def get_SrivatsanTrapnell2020(genespace=None, data='../../../data/', extdata='../../extdata/',
                     max_prct_ribo=0.05, max_prct_mito=0.05, min_ngenes_per_cell=250, 
                     min_cells_per_gene=10000, time=24, min_ncounts_per_obs=1000, 
                     n_top_variable_genes=1000):
    '''
    Sciplex3 dataset (~147 drugs with known targets)
    # TODO: compare to https://github.com/facebookresearch/CPA/blob/main/preprocessing/sciplex3.ipynb
    '''
    
    dataset = 'SrivatsanTrapnell2020_sciplex3'

    adata = load_scp_dataset(data, dataset)

    # filter gene vars by cell count 
    adata = adata[:, adata.var.ncells > min_cells_per_gene]

    # filter cells by gene count 
    adata = adata[adata.obs.ngenes > min_ngenes_per_cell]

    # filter high ribo 
    adata = adata[adata.obs.percent_ribo <= max_prct_ribo]

    # filter high mito 
    adata = adata[adata.obs.percent_mito <= max_prct_mito]

    # filter cells by ncounts 
    adata = adata[adata.obs.ncounts >= min_ncounts_per_obs]

    # ensure critical attribures are not nan 
    adata = adata[~adata.obs.time.isnull()]
    adata = adata[~adata.obs.cell_line.isnull()]

    # filter to time point 
    if time is not None: 
        adata = adata[adata.obs.time == time]

    # filter to human genes 
    adata = adata[:, adata.var.ensembl_id.str.contains('ENSG')]

    # merge var to get uniprot ids 
    uni2ens = pd.read_csv(extdata + '/omnipath_uniprot2ensembl.tsv', sep='\t').rename({'From':'uniprot', 'To':'ensembl_id'}, axis=1)
    uni2ens = uni2ens.dropna()
    uni2ens = uni2ens.assign(ensembl_id = lambda x: [xx.split('.')[0] for xx in x.ensembl_id]) # remove versioning
    uni2ens.uniprot = uni2ens.uniprot.values.astype(str)
    if genespace is not None: uni2ens = uni2ens[lambda x: x.uniprot.isin(genespace)]
    uni2ens = uni2ens.groupby('ensembl_id', as_index=False).first() # arbitrarily select first mapping to ensure 1:1 ; TODO: add allowance for duplicates ; NOTE: performing this after genespace filter ensures we select potential duplicate mappings that are in genespace

    adata.var = adata.var.merge(uni2ens, on='ensembl_id', how='left')
    adata = adata[:, ~adata.var.uniprot.isna()] # ensure all genes have an ensembl id

    # merge dep drug ids `pert_id`
    chmbl2dep = pd.read_csv(extdata + '/sciplex3_chembl2dep.csv')[['chembl-ID', 'pert_id']]
    chmbl2dep = chmbl2dep.groupby('chembl-ID', as_index=False).first() # arbitrarily select first mapping to ensure 1:1

    adata.obs = adata.obs.merge(chmbl2dep, on='chembl-ID', how='left')

    # Normalizing to median total counts
    sc.pp.normalize_total(adata)

    # log scale the counts data 
    sc.pp.log1p(adata)
    
    # remove low var genes and select most perturbed genes
    print('pre-filter # vars:', adata.var.shape[0])
    sc.pp.highly_variable_genes(adata, subset=True, min_disp=0.5, min_mean=0.0125)
    print('post var filter # vars:', adata.var.shape[0])
    if adata.var.shape[0] > n_top_variable_genes:
        adata = select_most_perturbed_genes(adata, N=n_top_variable_genes)
        print('post most perturbed filter # vars:', adata.var.shape[0])
    else: 
        print('skipping most perturbed filter (n_top_variable_genes > # vars)')

    sc.pp.scale(adata, zero_center=True) # zscored data

    ctrl_adata = adata[adata.obs.perturbation == 'control']

    drug_adata = adata[~adata.obs.pert_id.isna()]

    return drug_adata, ctrl_adata