import argparse
import numpy as np
import pandas as pd
import torch
import os
import sys
import warnings
from sklearn.model_selection import KFold

sys.path.append('../')
from gsnn_lib.proc.lincs.data_split import keys2sids
from gsnn_lib.proc.prism.utils import load_prism

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="You are using `torch.load` with `weights_only=False`.*"
)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default='../../../data/',
                        help="Path to raw data directory")
    
    parser.add_argument("--proc", type=str, default='../../proc/sciplex3/',
                        help="Path to processed data directory")

    parser.add_argument('--outer_k', type=int, default=5,
                        help='Number of outer folds for cross-validation')
    
    parser.add_argument('--min_obs_per_cond', type=int, default=100,
                        help='Minimum number of observations per condition for inclusion in the dataset')
    
    parser.add_argument('--val_prop', type=float, default=0.1,
                        help='Proportion of the training pool to become the validation set')

    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    print()
    print(args)
    print()

    args.out = os.path.join(args.proc, 'partitions')
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.out, 'data_split_args.log'), 'w') as f:
        f.write(str(args))

    # Load preprocessed data
    data = torch.load(os.path.join(args.proc, 'data.pt'))

    ctrlinfo = pd.read_csv(os.path.join(args.proc, 'ctrl_meta.csv'))
    druginfo = pd.read_csv(os.path.join(args.proc, 'drug_meta.csv'))

    np.random.seed(args.seed)

    print(f"\nCreating outer {args.outer_k}-fold splits with an internal train/val split (LINCS)...")


    ######################################################################
    # K-fold CV based on unique (cell, drug) pairs with a train/val split
    ######################################################################
    print('\tPerforming K-fold CV by unique (cell, drug) pairs...')

    # conditions 
    conditions = druginfo[['pert_id', 'cell_line', 'dose_value']].drop_duplicates()

    outer_kf = KFold(n_splits=args.outer_k, shuffle=True)
    
    splits = {i: {'pert':{'train': [], 'val': [], 'test': []}, 
                  'ctrl':{'train': [], 'val': [], 'test': []}} for i in range(args.outer_k)}

    for i, row in conditions.iterrows():

        drugcond = druginfo[lambda x: (x.cell_line == row.cell_line) & (x.pert_id == row.pert_id) & (x.dose_value == row.dose_value)]

        if len(drugcond) < args.min_obs_per_cond:
            print(f'\tcondition [[{row.pert_id} - {row.cell_line} - {row.dose_value}]] with {len(drugcond)} observations (less than {args.min_obs_per_cond}) - skipping')
        else:

            outer_kf = KFold(n_splits=args.outer_k, shuffle=True)
            for outer_fold_idx, (outer_train_ixs, outer_test_ixs) in enumerate(outer_kf.split(drugcond)):
                outer_train_ixs = drugcond.index[outer_train_ixs]
                outer_test_ixs  = drugcond.index[outer_test_ixs]

                val_ixs = np.random.choice(outer_train_ixs, size=int(np.floor(len(outer_train_ixs) * args.val_prop)), replace=False)
                train_ixs = np.setdiff1d(outer_train_ixs, val_ixs)

                splits[outer_fold_idx]['pert']['train'] += train_ixs.tolist()
                splits[outer_fold_idx]['pert']['val'] += val_ixs.tolist()
                splits[outer_fold_idx]['pert']['test'] += outer_test_ixs.tolist()

    # create control splits 

    for cell_line in ctrlinfo.cell_line.unique():

        ctrlcond = ctrlinfo[lambda x: x.cell_line == cell_line]

        outer_kf = KFold(n_splits=args.outer_k, shuffle=True)
        for outer_fold_idx, (outer_train_ixs, outer_test_ixs) in enumerate(outer_kf.split(ctrlcond)):
            outer_train_ixs = ctrlcond.index[outer_train_ixs]
            outer_test_ixs  = ctrlcond.index[outer_test_ixs]

            val_ixs = np.random.choice(outer_train_ixs, size=int(np.floor(len(outer_train_ixs) * args.val_prop)), replace=False)
            train_ixs = np.setdiff1d(outer_train_ixs, val_ixs)

            splits[outer_fold_idx]['ctrl']['train'] += train_ixs.tolist()
            splits[outer_fold_idx]['ctrl']['val'] += val_ixs.tolist()
            splits[outer_fold_idx]['ctrl']['test'] += outer_test_ixs.tolist()

    
    for i in range(args.outer_k):
        
        split_dict = splits[i]
        split_dir = os.path.join(args.out, f'split_{i}')
        os.makedirs(split_dir, exist_ok=True)
        torch.save(split_dict, os.path.join(split_dir, 'split_dict.pt'))

    print('# conditions:', len(conditions))

    print('\tSCIPLEX3 K-fold splits with train/val split saved successfully.')
