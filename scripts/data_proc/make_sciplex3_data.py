'''
examples: 

'''
import argparse 
import os 
import pandas as pd
import omnipath as op
import torch_geometric as pyg
import numpy as np
import torch
import h5py
import networkx as nx
import functools
from collections import deque
import copy

from gsnn_lib.proc.lincs.load_methyl import load_methyl
from gsnn_lib.proc.lincs.load_expr import load_expr
from gsnn_lib.proc.lincs.load_cnv import load_cnv
from gsnn_lib.proc.lincs.load_mut import load_mut
from gsnn_lib.proc.lincs.utils import get_x_drug_conc, get_geneid2uniprot
from gsnn_lib.proc.omnipath.utils import filter_func_nodes
from gsnn_lib.proc import omnipath
from gsnn_lib.proc import dti
from gsnn_lib.proc.sc.load import get_SrivatsanTrapnell2020


def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data",               type=str,               default='../../../data/',                      help="path to data directory")
    parser.add_argument("--out",                type=str,               default='../../output/sciplex3/',               help="path to data directory")
    parser.add_argument("--extdata",            type=str,               default='../../extdata/',                      help="path to data directory")
    parser.add_argument('--dti_sources',        nargs='+',              default=['clue', 'targetome', 'stitch'],              help='the databases to use for drug target prior knowledge [clue, stitch, targetome]')
    parser.add_argument("--filter_depth",       type=int,               default=10,                                 help="the depth to search for upstream drugs and downstream lincs in the node filter process")
    parser.add_argument("--n_genes",            type=int,               default=2000,                               help="selection of the top N high-variance RNA genes")
    parser.add_argument("--undirected",         action='store_true',    default=False,                              help="make all function edges undirected")
    parser.add_argument("--seed",               type=int,               default=0,                                  help="randomization seed")
    parser.add_argument('--val_prop',           type=float,             default=0.1,                                help='proportion of cells to assign to validation partition')
    parser.add_argument('--test_prop',          type=float,             default=0.1,                                help='proportion of cells to assign to test partition')
    parser.add_argument('--dose_eps_',          type=float,             default=1e-6,                               help='dose scaling epsilon [recommended: 1e-6]')
    args = parser.parse_args() 

    return args 

if __name__ == '__main__': 

    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print()

    if not os.path.exists(args.out): 
        print('making output directory...')
        os.makedirs(args.out, exist_ok=True)

    with open(f'{args.out}/args.log', 'w') as f: 
        f.write(str(args))
    
    # retrieve prior knowledge from omnipath
    func_names, func_df = omnipath.get_interactions(undirected=args.undirected)

    # get rna_uniprots
    func_df = func_df.assign(source_uni = [x.split('__')[1] for x in func_df.source])
    func_df = func_df.assign(target_uni = [x.split('__')[1] for x in func_df.target])
    func_df = func_df.assign(source_is_rna = ['RNA' in x for x in func_df.source])
    func_df = func_df.assign(target_is_rna = ['RNA' in x for x in func_df.target])
    rna_uniprots = np.unique(func_df.source_uni[func_df.source_is_rna].tolist() + func_df.target_uni[func_df.target_is_rna].tolist())

    ## load drug target interactions 
    targets = dti.get_interactions(extdata_dir=args.extdata, 
                                   sources=args.dti_sources, 
                                   func_names=func_names)

    ## load sc data ; Sciplex3
    print('loading and pre-processing single cell data')
    drug_adata, ctrl_adata = get_SrivatsanTrapnell2020(genespace=rna_uniprots, 
                                                       data=args.data,
                                                       extdata=args.extdata,
                                                        time=24, 
                                                        max_prct_mito=0, 
                                                        max_prct_ribo=0, 
                                                        min_ngenes_per_cell=100,
                                                        min_cells_per_gene=1000, 
                                                        min_ncounts_per_obs=100,
                                                        n_top_variable_genes=args.n_genes)

    # get overlapping drugs between sc exp and known targets 
    sc_drugs = drug_adata.obs.pert_id.unique()
    dti_drugs = targets.pert_id.unique()
    args.drugs = list(set(sc_drugs.tolist()).intersection(set(dti_drugs.tolist())))
     
    # "lincs" is deprecated here but we will use it in place of "outputs"
    args.lincs = drug_adata.var.uniprot.unique()

    # filter non-overlapping drugs 
    drug_adata = drug_adata[drug_adata.obs.pert_id.isin(args.drugs)]
    targets = targets[lambda x: x.pert_id.isin(args.drugs)]

    # add cell lines 
    args.cell_lines = drug_adata.obs.cell_line.unique()
    
    ######################################3
    ## START filter 
    ######################################3

    # filter nodes that are not downstream of a drug AND do not have downstream LINCS genes 
    # also filter targets that are no longer relevant 
    func_names, func_df, targets, args.drugs, args.lincs = filter_func_nodes(args, func_names, func_df, targets, args.lincs, args.drugs)

    func2idx = {f:i for i,f in enumerate(func_names)}
    func_df = func_df.assign(src_idx = [func2idx[f] for f in func_df.source.values])
    func_df = func_df.assign(dst_idx = [func2idx[f] for f in func_df.target.values])

    # need to remove filtered "output" vars from adata
    ctrl_adata = ctrl_adata[:, ctrl_adata.var.uniprot.isin(args.lincs)]
    drug_adata = drug_adata[:, drug_adata.var.uniprot.isin(args.lincs)]

    assert ctrl_adata.var.uniprot.unique().shape[0] == len(args.lincs), f'ctrl adata unique uniprots {ctrl_adata.var.uniprot.unique().shape[0]} are different than length of args.lincs {len(args.lincs)}'
    assert drug_adata.var.uniprot.unique().shape[0] == len(args.lincs), f'drug adata unique uniprots {drug_adata.var.uniprot.unique().shape[0]} are different than length of args.lincs {len(args.lincs)}'
    
    # need to remove filtered drugs from adata 
    drug_adata = drug_adata[drug_adata.obs.pert_id.isin(args.drugs)]

    assert drug_adata.obs.pert_id.unique().shape[0] == len(args.drugs), f'drug adata unique drugs {ctrl_adata.obs.pert_id.unique().shape[0]} are different than length of args.lincs {len(args.drugs)}'

    ######################################
    ## END filter 
    ######################################
    
    input_names = ['DRUG__' + d for d in args.drugs] + ['UNPERT_RNA__' + r for r in args.lincs]

    src=[]; dst=[]
    nde = 0 ; noe = 0 ; nce = 0
    for inp in input_names: 
        node, id = inp.split('__') 
        src_idx = input_names.index(inp)

        if node == 'DRUG': 
            targs = targets[lambda x: x.pert_id == id].target.values 
            for targ in targs: 
                dst_idx = func_names.index('PROTEIN__' + targ)
                src.append(src_idx); dst.append(dst_idx)
                nde+=1
                # question: should we add DTIs to complexes if they contain the target? 

        elif node == 'UNPERT_RNA': 
            if ('RNA__' + id) in func_names: 
                dst_idx = func_names.index('RNA__' + id)
                src.append(src_idx); dst.append(dst_idx)
                noe+=1
            if ('PROTEIN__' + id) in func_names: 
                dst_idx = func_names.index('PROTEIN__' + id)
                src.append(src_idx); dst.append(dst_idx)
                noe+=1

            # add edges to COMPLEXES that contain the respective id  
            # inefficient but should work 
            for dst_idx, fname in enumerate(func_names): 
                if ('COMPLEX' in fname) and (id in fname): 
                    src.append(src_idx); dst.append(dst_idx)
                    noe+=1
                    nce+=1 
        else:
            raise Exception()
        
    print('# drug edges', nde)
    print('# CTRL RNA edges', noe)
    print('# CTRL RNA edges to protein complexes', nce)
    input_edge_index = torch.stack((torch.tensor(src, dtype=torch.long), 
                                    torch.tensor(dst, dtype=torch.long)), dim=0)

    # `function` nodes 
    # These nodes are proteins and RNAs
    # remove duplicate edges 
    nn1 = func_df.shape[0]
    func_df = func_df[['source', 'target', 'src_idx', 'dst_idx']].drop_duplicates()
    print(f'# filtered duplicate edges: {nn1 - func_df.shape[0]}')
    func_edge_index = torch.stack((torch.tensor(func_df.src_idx.values, dtype=torch.long), 
                                   torch.tensor(func_df.dst_idx.values, dtype=torch.long)), dim=0)

    # `output` nodes 
    # These are the LINCS nodes, for which we are predicting. These are also fixed for prediction. 
    # assume lincs are provided as uniprot ids 
    lincs_space = ['RNAOUT__' + l for l in args.lincs]

    src = [] ; dst = []
    for linc in lincs_space: 
        node, id = linc.split('__') 
        src_idx = func_names.index('RNA__' + id)
        dst_idx = lincs_space.index(linc)
        src.append(src_idx); dst.append(dst_idx)

    output_edge_index = torch.stack((torch.tensor(src, dtype=torch.long), 
                                     torch.tensor(dst, dtype=torch.long)), dim=0)
    
    data = pyg.data.HeteroData() 

    # create data 
    data['edge_index_dict'] = {
        ('input',       'to',           'function')     : input_edge_index, 
        ('function',    'to',           'function')     : func_edge_index, 
        ('function',    'to',           'output')       : output_edge_index, 
    }

    data['node_names_dict'] = {'input':input_names,
                               'function':func_names,
                               'output':lincs_space}
    
    print('# `input` nodes:', len(data['node_names_dict']['input']))
    print('# `function` nodes:', len(data['node_names_dict']['function']))
    print('# `output` nodes:', len(data['node_names_dict']['output']))
    
    for key, edge_index in data['edge_index_dict'].items(): 
        print(f'# edges `{key}`: {edge_index.size(1)}')
    
    # Now lets create the inputs and outputs 

    ## inputs first 
    ## We can simplify this process by creating omic inputs (by cell line) and drug inputs (by drug + conc)
    ### drug-conc, cell line obs can then be created by simple addition of inputs (since nodes are independent)
    ### we can also save as sparse tensors to make reading into memory faster 

    inames = data.node_names_dict['input']
    onames = data.node_names_dict['output']
    data.X2Y0_idxs = torch.tensor([inames.index('UNPERT_RNA__' + ii.split('__')[1]) for ii in onames], dtype=torch.long)

    # CONTROL RNA INPUTS 
    ## we'll save ctrls to separate folder by cell line 
    os.makedirs(args.out + '/PROC/', exist_ok=True)

    inp2idx = {n:i for i,n in enumerate(data.node_names_dict['input'])}
    adata_idx_order = [inp2idx['UNPERT_RNA__' + rna] for rna in ctrl_adata.var.uniprot.values.tolist()]
    ctrl_adata.obs.reset_index(inplace=True)
    for i,row in ctrl_adata.obs.iterrows():
        print(f'writing sc UNPERTRUBED inputs to disk... {i}/{len(ctrl_adata.obs)}', end='\r') 
        x = torch.zeros((len(data.node_names_dict['input']),), dtype=torch.float32)
        x[adata_idx_order] = torch.tensor(ctrl_adata[i].X, dtype=torch.float32)
        x = x.detach().contiguous()
        torch.save(x, f'{args.out}/PROC/unperturbed_{i}_{row.pert_id}_{row.dose_value}_{row.cell_line}.pt')
    print()

    # BUG: this doesn't work for some reason - pickle instead
    #ctrl_adata.write_h5ad(args.out + '/ctrl_adata.h5')

    x_drug_dict = {} 
    # this will be more efficient then saving all unique drug concs 
    # x_drug_dict[drug](conc_um) -> x_drug_conc 
    for drug in args.drugs: 
        idx = input_names.index('DRUG__' + drug)
        f = functools.partial(get_x_drug_conc, idx=idx, N=len(input_names), eps=args.dose_eps_)
        x_drug_dict[drug] = f

    x_dict = {'cell_dict':None,
              'drug_dict':x_drug_dict}
    
    data['x_dict'] = x_dict
    
    ## outputs 
    # we will create the obs_space then save each one to file. 
    out2idx = {n:i for i,n in enumerate(data.node_names_dict['output'])}
    adata_idx_order2 = [out2idx['RNAOUT__' + rna] for rna in drug_adata.var.uniprot.values.tolist()]
    drug_adata.obs.reset_index(inplace=True)
    for i,row in drug_adata.obs.iterrows():
        print(f'writing sc outputs to disk... {i}/{len(drug_adata.obs)}', end='\r') 
        x = torch.zeros((len(data.node_names_dict['output']),), dtype=torch.float32)
        x[adata_idx_order2] = torch.tensor(drug_adata[i].X, dtype=torch.float32)
        x = x.detach().contiguous()
        torch.save(x, f'{args.out}/PROC/perturbed_{i}_{row.pert_id}_{row.dose_value}_{row.cell_line}.pt')
    print()

    torch.save(data, args.out + '/data.pt')

    # create data partitions (train, valid, test)
    print('creating data partitions...')
    n = len(drug_adata.obs)
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    n_val = int(args.val_prop * n)
    n_test = int(args.test_prop * n)
    n_train = n - n_val - n_test
    idxs_train = idxs[:n_train]
    idxs_val = idxs[n_train:n_train+n_val]
    idxs_test = idxs[n_train+n_val:]

    torch.save(torch.tensor(idxs_train, dtype=torch.long), args.out + '/train_idxs.pt')
    torch.save(torch.tensor(idxs_val, dtype=torch.long), args.out + '/val_idxs.pt')
    torch.save(torch.tensor(idxs_test, dtype=torch.long), args.out + '/test_idxs.pt')

    with open(f'{args.out}/make_data_completed_successfully.flag', 'w') as f: f.write(':)')


