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
    
    parser.add_argument("--proc", type=str, default='../../proc/lincs/',
                        help="Path to processed data directory")
    
    parser.add_argument("--hold_out", type=str, default='cell-drug',
                        help="How to split for cross-validation; options: 'cell' or 'cell-drug'")

    parser.add_argument('--outer_k', type=int, default=5,
                        help='Number of outer folds for cross-validation')
    
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

    condinfo = pd.read_csv(os.path.join(args.proc, 'conditions_meta.csv'), sep=',')
    cond_ids = condinfo.condition_id.values

    np.random.seed(args.seed)

    print("\nCreating outer K-fold splits with an internal train/val split (LINCS)...")

    if args.hold_out == 'cell':
        ######################################################################
        # K-fold CV based on unique cell lines with a train/val split
        ######################################################################
        print('\tPerforming K-fold CV by unique cell line...')

        unique_cells = np.array(data.cellspace)
        outer_kf = KFold(n_splits=args.outer_k, shuffle=True, random_state=args.seed)
        outer_folds = list(outer_kf.split(unique_cells))

        for outer_fold_idx, (outer_train_ixs, outer_test_ixs) in enumerate(outer_folds):
            # Outer fold test cells
            test_cells = unique_cells[outer_test_ixs]
            train_cells = unique_cells[outer_train_ixs]

            # Observations corresponding to these cells
            train_obs = condinfo[lambda x: x.cell_iname.isin(train_cells)].condition_id.values
            test_obs = condinfo[lambda x: x.cell_iname.isin(test_cells)].condition_id.values

            # Check for test-only drugs
            test_only_drugs = set(condinfo[lambda x: x.cell_iname.isin(test_cells)].pert_id.unique()) - \
                              set(condinfo[lambda x: x.cell_iname.isin(train_cells)].pert_id.unique())

            if len(test_only_drugs) > 0:
                print(f'\tWARNING Outer Fold {outer_fold_idx}: {len(test_only_drugs)} drugs only appear in test set.')
                test_obs = condinfo[lambda x: (x.cell_iname.isin(test_cells)) & (~x.pert_id.isin(test_only_drugs))].condition_id.values

            # From the training pool, create a validation split
            val_size = int(np.floor(len(train_cells) * args.val_prop))
            val_ixs = np.random.choice(len(train_cells), size=val_size, replace=False)
            val_cells = train_cells[val_ixs]

            # The remaining training cells
            train_mask = np.ones(len(train_cells), dtype=bool)
            train_mask[val_ixs] = False
            final_train_cells = train_cells[train_mask]

            # Get observations for train/val
            val_obs = condinfo[lambda x: x.cell_iname.isin(val_cells)].condition_id.values
            final_train_obs = condinfo[lambda x: x.cell_iname.isin(final_train_cells)].condition_id.values

            print(f'\tOuter Fold {outer_fold_idx}:')
            print(f'\t# final train cells (# obs): {len(final_train_cells)} ({len(final_train_obs)})')
            print(f'\t# val cells   (# obs): {len(val_cells)} ({len(val_obs)})')
            print(f'\t# test cells  (# obs): {len(test_cells)} ({len(test_obs)})')
            print()

            # Checks
            assert len(set(final_train_obs).intersection(set(val_obs))) == 0, 'Train/val overlap!'
            assert len(set(final_train_obs).intersection(set(test_obs))) == 0, 'Train/test overlap!'
            assert len(set(val_obs).intersection(set(test_obs))) == 0, 'Val/test overlap!'

            # Save sets for this fold
            splits_dict = {
                "train_cells": final_train_cells,
                "val_cells": val_cells,
                "test_cells": test_cells,
                "train_obs": final_train_obs,
                "val_obs": val_obs,
                "test_obs": test_obs
            }
            torch.save(splits_dict, os.path.join(args.out, f'fold_{outer_fold_idx}_splits.pt'))

    elif args.hold_out == 'cell-drug':
        ######################################################################
        # K-fold CV based on unique (cell, drug) pairs with a train/val split
        ######################################################################
        print('\tPerforming K-fold CV by unique (cell, drug) pairs...')

        condinfo['key'] = list(zip(condinfo.cell_iname, condinfo.pert_id))
        unique_keys = condinfo['key'].unique()

        outer_kf = KFold(n_splits=args.outer_k, shuffle=True, random_state=args.seed)
        outer_folds = list(outer_kf.split(unique_keys))

        for outer_fold_idx, (outer_train_ixs, outer_test_ixs) in enumerate(outer_folds):
            train_keys = unique_keys[outer_train_ixs]
            test_keys = unique_keys[outer_test_ixs]

            train_obs = condinfo[condinfo.key.isin(train_keys)].condition_id.values
            test_obs = condinfo[condinfo.key.isin(test_keys)].condition_id.values

            # Create a validation split from the training keys
            val_size = int(np.floor(len(train_keys) * args.val_prop))
            val_ixs = np.random.choice(len(train_keys), size=val_size, replace=False)
            val_keys = train_keys[val_ixs]

            # Remaining keys
            train_mask = np.ones(len(train_keys), dtype=bool)
            train_mask[val_ixs] = False
            final_train_keys = train_keys[train_mask]

            val_obs = condinfo[condinfo.key.isin(val_keys)].condition_id.values
            final_train_obs = condinfo[condinfo.key.isin(final_train_keys)].condition_id.values

            print(f'\tOuter Fold {outer_fold_idx}:')
            print(f'\t# final train keys (# obs): {len(final_train_keys)} ({len(final_train_obs)})')
            print(f'\t# val keys   (# obs): {len(val_keys)} ({len(val_obs)})')
            print(f'\t# test keys  (# obs): {len(test_keys)} ({len(test_obs)})')
            print()

            # add check for test only drugs 
            test_only_drugs = set(condinfo[condinfo.key.isin(test_keys)].pert_id.unique()) - \
                              set(condinfo[condinfo.key.isin(final_train_keys)].pert_id.unique())
            
            if len(test_only_drugs) > 0:
                print(f'\tWARNING Outer Fold {outer_fold_idx}: {len(test_only_drugs)} drugs only appear in test set. These drugs will be remove from test set.')
                test_obs = condinfo[(condinfo.key.isin(test_keys)) & (~condinfo.pert_id.isin(test_only_drugs))].condition_id.values

            # Checks
            assert len(set(final_train_obs).intersection(set(val_obs))) == 0, 'Train/val overlap!'
            assert len(set(final_train_obs).intersection(set(test_obs))) == 0, 'Train/test overlap!'
            assert len(set(val_obs).intersection(set(test_obs))) == 0, 'Val/test overlap!'

            # Save sets for this fold
            splits_dict = {
                "train_keys": final_train_keys,
                "val_keys": val_keys,
                "test_keys": test_keys,
                "train_obs": final_train_obs,
                "val_obs": val_obs,
                "test_obs": test_obs
            }
            torch.save(splits_dict, os.path.join(args.out, f'fold_{outer_fold_idx}_lincs_splits.pt'))

    else:
        raise ValueError('Unrecognized `hold_out` argument; options: cell, cell-drug')

    print('\tLINCS K-fold splits with train/val split saved successfully.')

'''
########################################################################
# Now create PRISM splits in a separate loop, based on previously saved
# LINCS splits per fold.
########################################################################

print('\nCreating PRISM train/test/val splits for each fold...')

# Load PRISM data once
print('loading PRISM data...')
prism = load_prism(args.data, cellspace=data.cellspace, drugspace=data.drugspace)
prism['key'] = list(zip(prism.cell_iname, prism.pert_id))
# prism columns: ['prism_id', 'pert_id', 'conc_um', 'cell_iname', 'cell_viab', 'log_fold_change']

# For each fold, load the LINCS splits and derive PRISM splits
for outer_fold_idx in range(args.outer_k):
    lincs_splits = torch.load(os.path.join(args.out, f'fold_{outer_fold_idx}_lincs_splits.pt'))

    if args.hold_out == 'cell':
        # LINCS splits are based on cells
        final_train_cells = lincs_splits['train_cells']
        val_cells = lincs_splits['val_cells']
        test_cells = lincs_splits['test_cells']

        train_obs2 = prism[lambda x: x.cell_iname.isin(final_train_cells)].prism_id.values
        val_obs2 = prism[lambda x: x.cell_iname.isin(val_cells)].prism_id.values
        test_obs2 = prism[lambda x: x.cell_iname.isin(test_cells)].prism_id.values

    elif args.hold_out == 'cell-drug':
        # LINCS splits are based on keys
        final_train_keys = lincs_splits['train_keys']
        val_keys = lincs_splits['val_keys']
        test_keys = lincs_splits['test_keys']

        val_obs2 = prism[lambda x: x.key.isin(val_keys)].prism_id.values
        test_obs2 = prism[lambda x: x.key.isin(test_keys)].prism_id.values
        
        train_obs2 = prism[lambda x: (~x.key.isin(val_keys)) & (~x.key.isin(test_keys))].prism_id.values
        #train_obs2 = prism[lambda x: x.key.isin(final_train_keys)].prism_id.values

    else:
        raise ValueError('Unrecognized `hold_out` argument; options: cell, cell-drug')

    # Save PRISM sets for this fold
    prism_splits = {
        "train_obs": train_obs2,
        "val_obs": val_obs2,
        "test_obs": test_obs2
    }
    torch.save(prism_splits, os.path.join(args.out, f'fold_{outer_fold_idx}_prism_splits.pt'))

    print(f'PRISM splits for fold {outer_fold_idx}:')
    print(f'\ttrain: {len(train_obs2)}')
    print(f'\tval: {len(val_obs2)}')
    print(f'\ttest: {len(test_obs2)}')
    print()

print('PRISM splits created successfully.')
'''
