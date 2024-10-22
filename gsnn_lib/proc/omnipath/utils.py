
from gsnn_lib.proc.omnipath.subset import subset_graph
import networkx as nx 
import numpy as np

def build_nx(func_df, targets, outputs): 
    G = nx.DiGraph()

    # function -> function 
    for i,edge in func_df.iterrows(): 
        G.add_edge(edge.source, edge.target)

    # drug -> function
    for i,edge in targets.iterrows(): 
        if ('PROTEIN__' + edge.target) in G: 
            G.add_edge('DRUG__' + edge.pert_id, 'PROTEIN__' + edge.target)
        else: 
            print(f'warning: {edge.target} is not present in graph, this DTI will not be added.')

    # function -> output edges
    for out in outputs: 
        # add the edge even if the RNA doesn't exist; will get filtered in next step
        G.add_edge('RNA__' + out, 'LINCS__' + out)

    return G

def filter_func_nodes(args, func_names, func_df, targets, lincs, drugs): 
    print('filtering function nodes...')
    G = build_nx(func_df, targets, lincs)
    
    subgraph = subset_graph(G, args.filter_depth, roots=['DRUG__' + x for x in args.drugs], leafs=['LINCS__' + x for x in args.lincs], verbose=True)
    nodes = list(subgraph.nodes())

    func_mask = np.array([n in nodes for n in func_names])
    drug_mask = np.array([('DRUG__' + n) in nodes for n in drugs])
    linc_mask = np.array([('LINCS__' + n) in nodes for n in lincs])

    print(f'function nodes retained: {(1.*func_mask).sum()}/{len(func_mask)}')
    print(f'drug nodes retained: {(1.*drug_mask).sum()}/{len(drug_mask)}')
    print(f'lincs nodes retained: {(1.*linc_mask).sum()}/{len(linc_mask)}')

    func_names = np.array(func_names)[func_mask].tolist()
    drugs = np.array(drugs)[drug_mask].tolist() 
    lincs = np.array(lincs)[linc_mask].tolist()

    func_df = func_df[lambda x: x.source.isin(func_names) & (x.target.isin(func_names))]
    targets = targets[lambda x: x.target_name.isin(func_names)]

    return func_names, func_df, targets, drugs, lincs
