
import pandas as pd 
import numpy as np 


def get_interactions(extdata_dir, sources, func_names): 
    
    # filter targets to prot space
    targets = pd.read_csv(extdata_dir + '/processed_targets.csv')

    # filter to dataset resources 
    print('DTI sources: ', sources)
    inclusion = np.zeros((targets.shape[0]))
    if 'clue' in sources:
        inclusion += 1.*targets.in_clue.values 
    else: 
        print('clue DTIs will not be included in prior knowledge')
    if 'targetome' in sources:
        inclusion += 1.*targets.in_targetome.values 
    else: 
        print('targetome DTIs will not be included in prior knowledge')
    if 'stitch' in sources:
        inclusion += 1.*targets.in_stitch.values 
    else: 
        print('stitch DTIs will not be included in prior knowledge')
    targets = targets[inclusion > 0] # select targets that are in our included datasets
    print(f'DTIs retained: {np.sum(1.*(inclusion > 0))}/{len(inclusion)}')

    targets = targets.assign(target_name = ['PROTEIN__' + x for x in targets.target])
    targets = targets[lambda x: x.target_name.isin(func_names)]
    
    return targets