
def get_drug_subgraphs(args, input_names, func_names, output_names, func_df, targets): 
    '''
    NOTE: using subgraphs isn't identical to using the full graph since some omic nodes may be omitted that can play a role.
    '''

    print('computing drug-specific subgraphs...')
    G = build_nx(func_df, targets, args.lincs)

    omic_nodes = [x for x in input_names if 'DRUG__' not in x]

    subgraph_dict = {}
    distance_dicts = ({}, {})
    for i, drug in enumerate(args.drugs): 
        print(f'progress: {i}/{len(args.drugs)}', end='\r')
        subgraph, distance_dicts = subset_graph(G, args.filter_depth, roots=['DRUG__' + drug], leafs=['LINCS__' + x for x in args.lincs], verbose=False, distance_dicts=distance_dicts, return_dicts=True)
        inp2idx = {node:i for i,node in enumerate(input_names)}
        fun2idx = {node:i for i,node in enumerate(func_names)}
        out2idx = {node:i for i,node in enumerate(output_names)}

        keep_omics = []
        for omic in omic_nodes: 
            typ, uni = omic.split('__')
            if (('RNA__' + uni) in subgraph) or (('PROTEIN__' + uni) in subgraph): 
                keep_omics.append(omic)
        
        subgraph_dict['DRUG__' + drug] = (torch.tensor([inp2idx[x] for x in input_names if x in subgraph] + [inp2idx[x] for x in keep_omics], dtype=torch.long),
                                          torch.tensor([fun2idx[x] for x in func_names if x in subgraph], dtype=torch.long),
                                          torch.tensor([out2idx[x] for x in output_names if x in subgraph], dtype=torch.long))
    return subgraph_dict