'''
examples: 

# all lines, all drugs, all lincs [no methyl]
python make_lincs_data.py --data ../../../data/ --out ../../proc/lincs/ --dti_sources targetome
python create_data_splits.py --data ../../../data/ --proc ../../proc/lincs/
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

def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data",               type=str,               default='../../../data/',                      help="path to data directory")
    parser.add_argument("--out",                type=str,               default='../../proc/lincs/',                help="path to data directory")
    parser.add_argument("--extdata",            type=str,               default='../../extdata/',                   help="path to data directory")
    parser.add_argument('--feature_space',      nargs='+',              default=['landmark'],                       help='lincs feature space [landmark, best-inferred, inferred]')
    parser.add_argument('--dti_sources',        nargs='+',              default=['clue', 'targetome'],              help='the databases to use for drug target prior knowledge [clue, stitch, targetome]')
    parser.add_argument('--drugs',              nargs='+',              default=['none'],                           help='list of drugs to include in the graph')
    parser.add_argument('--lines',              nargs='+',              default=['none'],                           help='list of cell lines to include in the graph')
    parser.add_argument('--lincs',              nargs='+',              default=['none'],                           help='list of lincs genes to include in the graph')
    parser.add_argument('--omics',              nargs='+',              default=['mut', 'methyl', 'expr', 'cnv'],   help='list of lincs genes to include in the graph')
    parser.add_argument('--omics_q_filter',     type=float,             default=0.25,                               help='features with variance in the `q` quantile will be removed (remove low variance features)')
    parser.add_argument("--time",               type=float,             default=24.,                                help="the time point to predict expression changes for")
    parser.add_argument("--filter_depth",       type=int,               default=10,                                 help="the depth to search for upstream drugs and downstream lincs in the node filter process")
    parser.add_argument("--min_obs_per_drug",   type=int,               default=100,                                help="if `--drugs` is None, then this will be the minimum number of obs to be included as drug candidate")
    parser.add_argument("--undirected",         action='store_true',    default=False,                              help="make all function edges undirected")
    parser.add_argument("--N_false_dti_edges",  type=int,               default=0,                                  help="number of false drug-> function edges to add to the graph")
    parser.add_argument('--dose_epsilon',       type=float,             default=1e-6,                               help='scaling parameter for dose transformation')
    parser.add_argument('--norm',               type=str,               default='zscore',                           help='normalization method for omics [zscore, minmax, none]')

    args = parser.parse_args() 

    if args.drugs[0] == 'none': args.drugs = None
    if args.lines[0] == 'none': args.lines = None
    if args.lincs[0] == 'none': args.lincs = None
    
    for f in args.feature_space: 
        if f not in ['landmark', 'best-inferred', 'inferred']: 
            raise ValueError(f'unrecognized feature space value: {f} [expects one of: landmark, best-inferred, inferred]')
        
    for f in args.omics: 
        if f not in ['mut', 'methyl', 'expr', 'cnv']: 
            raise ValueError(f'unrecognized omic')

    args.feature_space = [x.replace('-', ' ') for x in args.feature_space]

    return args 

if __name__ == '__main__': 

    args = get_args()
    print(args)
    print()

    if not os.path.exists(args.out): 
        print('making output directory...')
        os.makedirs(args.out, exist_ok=True)

    with open(f'{args.out}/args.log', 'w') as f: 
        f.write(str(args))
    

    # retrieve prior knowledge from omnipath
    func_names, func_df = omnipath.get_interactions(undirected=args.undirected)

    ##################################
    # load omics here
    ##################################
    _omics = {} 
    if ('methyl' in args.omics) or (args.omics is None): 
        methyl  = load_methyl(path=args.data, extpath=args.extdata) ; print('\t\tmethyl loaded.')
        _omics['methyl'] = {'df':methyl}
    if ('expr' in args.omics) or (args.omics is None): 
        expr    = load_expr(path=args.data, extpath=args.extdata, zscore=False, clip_val=10) ; print('\t\texpr loaded.')
        _omics['expr'] = {'df':expr}
    if ('cnv' in args.omics) or (args.omics is None): 
        cnv     = load_cnv(path=args.data, extpath=args.extdata) ; print('\t\tcnv loaded.')
        _omics['cnv'] = {'df':cnv}
    if ('mut' in args.omics) or (args.omics is None): 
        mut     = load_mut(path=args.data, extpath=args.extdata) ; print('\t\tmut loaded.')
        _omics['mut'] = {'df':mut}

    line_candidates = None
    for om, di in _omics.items(): 
        line_candidates = set(di['df'].index.tolist()) if line_candidates is None else line_candidates.intersection(set(di['df'].index.tolist()))
    print('# of cell line candidates (from omics):', len(line_candidates))

    if args.lines is None: 
        siginfo = pd.read_csv(f'{args.data}/siginfo_beta.txt', sep='\t', low_memory=False) 
        args.lines = list(set(siginfo.cell_iname.unique().tolist()).intersection(line_candidates))
    else: 
        args.lines = list(set(args.lines).intersection(line_candidates))
    print('# cell lines to include:', len(args.lines))
    
    ## load drug target interactions 
    targets = dti.get_interactions(extdata_dir=args.extdata, 
                                   sources=args.dti_sources, 
                                   func_names=func_names)

    if args.drugs is not None: 
        targets = targets[lambda x: x.pert_id.isin(args.drugs)]
    else: 
        siginfo = pd.read_csv(f'{args.data}/siginfo_beta.txt', sep='\t', low_memory=False)
        siginfo = siginfo[lambda x: (x.pert_type.isin(['trt_cp'])) 
                        & (x.qc_pass == 1.)                         # passes QC
                        & (np.isfinite(x.pert_dose.values))
                        & (x.pert_time == args.time)
                        & (x.cell_iname.isin(args.lines))]
        drugs1 = np.unique(siginfo.groupby('pert_id').count()[['sig_id']][lambda x: x.sig_id >= args.min_obs_per_drug].index.tolist())
        drugs2 = targets.pert_id.unique() 
        args.drugs = list(set(drugs1).intersection(set(drugs2)))
    print('# of drug candidates:', len(args.drugs))
 

    if args.lincs is None: 
        gene_info = pd.read_csv(f'{args.data}/geneinfo_beta.txt', sep='\t')
        gene2uni = get_geneid2uniprot(args)
        gene_ids = gene_info[lambda x: x.feature_space.isin(args.feature_space)].gene_id.values.tolist()
        args.lincs = [gene2uni[x] for x in gene_ids if x in gene2uni]
        print(f'# lincs outputs to include [feature space={args.feature_space}]: {len(args.lincs)}')

    ######################################3
    ## START filter 
    ######################################3

    # filter nodes that are not downstream of a drug AND do not have downstream LINCS genes 
    # also filter targets that are no longer relevant 
    func_names, func_df, targets, args.drugs, args.lincs = filter_func_nodes(args, func_names, func_df, targets, args.lincs, args.drugs)

    func2idx = {f:i for i,f in enumerate(func_names)}
    func_df = func_df.assign(src_idx = [func2idx[f] for f in func_df.source.values])
    func_df = func_df.assign(dst_idx = [func2idx[f] for f in func_df.target.values])
    
    ######################################3
    ## END filter 
    ######################################3
    
    # NOTE: Because we don't need to include an omic for every node, we can similarly filter to omics that have good variance
    omics = {}
    omic_space = []
    for omic, di in _omics.items():  
        df = di['df']
        if args.lines is not None: df = df[lambda x: x.index.isin(args.lines)]                     # filter to cellspace
        genes = df.columns[df.std(axis=0) >= np.quantile(df.std(axis=0), q=args.omics_q_filter)]   # filter low variance features
        genes = [g for g in genes if (g in [x.split('__')[1] for x in func_names])]                # filter to prot/rna space
        df = df[genes]
        omics[omic] = {'df':df, 'genes':genes}
        omic_space += [omic.upper() + '__' + g for g in genes]

    print('# of omic inputs:', len(omic_space))

    # ADD FALSE DRUG -> PROTEIN EDGES
    targets = targets.assign(false_edge=False)
    if args.N_false_dti_edges > 0: 
        dcands = [d for d in args.drugs]
        pcands = [p.split('__')[1] for p in func_names if 'PROTEIN' in p]
        pert_id_ = np.random.choice(dcands, size=args.N_false_dti_edges)
        target_ = np.random.choice(pcands, size=args.N_false_dti_edges)

        targets = pd.concat([targets, pd.DataFrame({'pert_id':pert_id_, 
                                                    'target':target_, 
                                                    'false_edge':[True]*len(pert_id_)})], axis=0)

    # `input` nodes 
    #  These nodes are drugs and omic features and are fixed for the optimization procedure. 
    drug_space = args.drugs
    input_names = ['DRUG__' + d for d in drug_space] + omic_space

    src=[]; dst=[] ; true_input_edge_mask = []
    nde = 0 ; noe = 0 ; nce = 0
    for inp in input_names: 
        node, id = inp.split('__') 
        src_idx = input_names.index(inp)

        if node == 'DRUG': 
            targs = targets[lambda x: x.pert_id == id]
            for i,row in targs.iterrows():
                targ = row.target 
                dst_idx = func_names.index('PROTEIN__' + targ)
                src.append(src_idx); dst.append(dst_idx)
                nde+=1
                # question: should we add DTIs to complexes if they contain the target? 

                if row.false_edge: 
                    true_input_edge_mask.append(False)
                else: 
                    true_input_edge_mask.append(True)

        elif node in ['MUT', 'METHYL', 'CNV', 'EXPR']: 

            if ('RNA__' + id) in func_names: 
                true_input_edge_mask.append(True)
                dst_idx = func_names.index('RNA__' + id)
                src.append(src_idx); dst.append(dst_idx)
                noe+=1
            if ('PROTEIN__' + id) in func_names: 
                true_input_edge_mask.append(True)
                # add edges to COMPLEXES that contain the respective id 
                dst_idx = func_names.index('PROTEIN__' + id)
                src.append(src_idx); dst.append(dst_idx)
                noe+=1

            # add edges to COMPLEXES that contain the respective id  
            # inefficient but should work 
            for dst_idx, fname in enumerate(func_names): 
                if ('COMPLEX' in fname) and (id in fname): 
                    true_input_edge_mask.append(True)
                    src.append(src_idx); dst.append(dst_idx)
                    noe+=1
                    nce+=1 
        else:
            raise Exception()
        
    assert len(src) == len(dst) == len(true_input_edge_mask), 'input edge index mismatch'
        
    print('# drug edges', nde)
    print('# omic edges', noe)
    print('# omic edges to protein complexes', nce)
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
    lincs_space = ['LINCS__' + l for l in args.lincs]
    #print('LINCS_SPACE:', lincs_space)

    src = [] ; dst = []
    for linc in lincs_space: 
        node, id = linc.split('__') 
        src_idx = func_names.index('RNA__' + id)
        dst_idx = lincs_space.index(linc)
        src.append(src_idx); dst.append(dst_idx)

    output_edge_index = torch.stack((torch.tensor(src, dtype=torch.long), 
                                     torch.tensor(dst, dtype=torch.long)), dim=0)
    
    data = pyg.data.HeteroData() 
    data.true_input_edge_mask = torch.tensor(true_input_edge_mask, dtype=torch.bool)

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

    # Need this if args.lines is None
    siginfo = pd.read_csv(f'{args.data}/siginfo_beta.txt', sep='\t', low_memory=False)

    siginfo = siginfo[lambda x: (x.pert_type.isin(['trt_cp'])) 
                        & (x.qc_pass == 1.)                         # passes QC
                        & (x.pert_id.isin(args.drugs))
                        & (np.isfinite(x.pert_dose.values))
                        & (x.pert_time == args.time)
                        & (x.cell_iname.isin(args.lines))]

    data.cellspace = args.lines
    data.drugspace = drug_space

    x_cell_dict = {} 
    for line in args.lines: 
        x = torch.zeros((len(input_names)), dtype=torch.float32)
        for idx, inp in enumerate(input_names): 
            node, id = inp.split('__')
            if node in ['MUT', 'METHYL', 'CNV', 'EXPR']: 
                xx = omics[node.lower()]['df'].loc[line][id]
                x[idx] = xx
        x_cell_dict[line] = x.to_sparse() 

    x_drug_dict = {} 
    # this will be more efficient then saving all unique drug concs 
    # x_drug_dict[drug](conc_um) -> x_drug_conc 
    for drug in drug_space: 
        idx = input_names.index('DRUG__' + drug)
        f = functools.partial(get_x_drug_conc, idx=idx, N=len(input_names), eps=args.dose_epsilon)
        #f = lambda x: get_x_drug_conc(x, idx=idx, N=len(input_names))
        x_drug_dict[drug] = f

    x_dict = {'cell_dict':x_cell_dict,
              'drug_dict':x_drug_dict}
    
    data['x_dict'] = x_dict
    
    ## outputs 
    # we will create the obs_space then save each one to file. 
     
    hdf_cp            = h5py.File(args.data + '/level5_beta_trt_cp_n720216x12328.gctx')
    dataset_cp        = hdf_cp['0']['DATA']['0']['matrix']
    col_cp            = np.array(hdf_cp['0']['META']['COL']['id'][...].astype('str'))       # lincs sample ids 
    row_cp            = hdf_cp['0']['META']['ROW']['id'][...].astype(int)                   # gene ids 

    print()
    print('# obs grouped by cell line: ')
    print(siginfo.groupby('cell_iname').count()[['sig_id']].sort_values(by='sig_id', ascending=False).head(25))
    print()

    print()
    print('# obs grouped by drug: ')
    print(siginfo.groupby('pert_id').count()[['sig_id']].sort_values(by='sig_id', ascending=False).head(25))
    print()

    print()
    print('# obs grouped by dose: ')
    print(siginfo.groupby('pert_dose').count()[['sig_id']].sort_values(by='sig_id', ascending=False).head(10))
    print()

    sig_ids = np.unique(siginfo.sig_id.values)

    os.makedirs(args.out, exist_ok=True)
    torch.save(sig_ids, args.out + '/sig_ids.pt')

    # convert gene ids to uniprot 
    gene2uni = get_geneid2uniprot(args)

    print('create hdf row idx...')
    row_idxs = []
    for i,gid in enumerate(row_cp): 
        if gid in gene2uni: 
            uid = gene2uni[gid]
            if ("LINCS__" + uid) in lincs_space: 
                row_idxs.append(i)
    row_idxs = np.array(row_idxs).ravel()

    assert len(row_idxs) == len(lincs_space), 'the lincs hdf row idxs have a different number of items as the provided lincs space; probably error with geneid -> uniprot conversion'

    print('create hdf col idxs...')
    sigid2idx = {sid:i for i,sid in enumerate(col_cp)}
    col_idxs = np.array([sigid2idx[sid] for sid in sig_ids]).ravel()

    # will be ordered as (sig_ids, lincs_space)
    print('loading hdf to memory...')
    dataset = dataset_cp[col_idxs, :][:, row_idxs]

    print('total number of lincs observations (sig_ids):', len(sig_ids))

    # TODO: I want to average the data within condition ( cell, drug, dose )
    # we will need to get the sig_ids for each condition, load the data into memory, average it, assign a new id (condition)
    # we will need to also make a new siginfo file that maps condition to set of sigids as well as the condition parameters themselves 
    # this files should be saved to `out`
    #conditions = siginfo[['cell_iname', 'pert_id', 'pert_dose', 'sig_id']].groupby(['cell_iname', 'pert_id', 'pert_dose']).agg(lambda x: list(x)).reset_index()
    # Average the data within each condition (cell, drug, dose)
    # First, group siginfo by condition
    conditions = (
        siginfo[['cell_iname', 'pert_id', 'pert_dose', 'sig_id', 'pert_time']]
        .groupby(['cell_iname', 'pert_id', 'pert_dose', 'pert_time'])
        .agg(list)
        .reset_index()
    )

    print('# of conditions (drug, cell-line, dose, time):', len(conditions))

    # For each condition, average the replicates
    new_sig_ids = []
    new_dataset = []
    condition_map = []  # store the original sig_ids for reference

    df = {'cell_iname':[], 'pert_id':[], 'pert_dose':[], 'pert_time':[], 'sig_ids':[], 'condition_id':[]}
    for i, row in conditions.iterrows():
        print('averaging conditions... ', i, ' / ', len(conditions), end='\r')
        original_sigids = row['sig_id']
        # Get the dataset rows corresponding to these sig_ids
        idxs = [np.where(sig_ids == s)[0][0] for s in original_sigids]
        cond_data = dataset[idxs, :]

        # Average across replicates
        avg_data = cond_data.mean(axis=0)

        # Create a new condition_id
        condition_id = f"{row['cell_iname']}__{row['pert_id']}__{row['pert_dose']}__{row['pert_time']}"

        new_sig_ids.append(condition_id)
        new_dataset.append(avg_data)
        condition_map.append(original_sigids)

        df['cell_iname'].append(row['cell_iname'])
        df['pert_id'].append(row['pert_id'])
        df['pert_dose'].append(row['pert_dose'])
        df['pert_time'].append(row['pert_time'])
        df['condition_id'].append(condition_id)
        df['sig_ids'].append(original_sigids)

    print()

    df = pd.DataFrame(df)

    new_dataset = np.vstack(new_dataset)

    # Update sig_ids and dataset to represent conditions rather than individual signatures
    sig_ids = np.array(new_sig_ids)
    dataset = new_dataset

    # Save the condition mapping for future reference
    conditions['condition_id'] = sig_ids
    conditions['original_sig_ids'] = condition_map
    conditions.to_csv(args.out + '/condition_mapping.csv', index=False)
    df.to_csv(args.out + '/conditions_meta.csv', index=False)

    # Perform normalization on the averaged dataset
    if args.norm == 'zscore':
        # NOTE: this normalizes across all LINCS genes (not within gene)
        mu = dataset.mean()
        std = dataset.std()
        transform = lambda x: (x - mu) / (std + 1e-8)
        transform_params = {'mu':mu, 'std':std, 'method':'zscore'}
    elif args.norm == 'minmax':
        # NODE: this normalizes within gene 
        min_ = dataset.min(axis=0)
        max_ = dataset.max(axis=0)
        transform = lambda x: (x - min_) / (max_ - min_ + 1e-8)
        transform_params = {'min':min_, 'max':max_, 'method':'minmax'}
    elif args.norm == 'none':
        transform = lambda x: x
        transform_params = {'method':'none'}

    torch.save(transform_params, args.out + '/obs_transform_params.pt')

    print('saving averaged obs to disk...')
    os.makedirs(args.out + '/obs/', exist_ok=True)
    for i, (cond_id, y) in enumerate(zip(sig_ids, dataset)):
        print(f'saving to disk... {i}/{len(sig_ids)}', end='\r')
        y = transform(y)
        y = torch.tensor(y, dtype=torch.float32)
        torch.save(y, args.out + '/obs/' + cond_id + '.pt')
    print('# LINCS observations (averaged conditions):', i+1)

    print('saving data object...')
    torch.save(data, args.out + '/data.pt')

    with open(f'{args.out}/make_data_completed_successfully.flag', 'w') as f: f.write(':)')

