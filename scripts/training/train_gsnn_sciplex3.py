import torch 
import argparse 
import uuid
import os 
import time
import numpy as np 
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, roc_auc_score
from sklearn import metrics
import pandas as pd

import sys 
from gsnn.models import utils 
from gsnn_lib.proc.lincs.utils import get_x_drug_conc           # required to unpickle data 

from gsnn_lib.data.scSampler import scSampler
from gsnn.ot.NOT import NOT
from gsnn.ot.SHD import SHD
from gsnn.ot.OTICNN import OTICNN
from gsnn.ot.utils import eval

from gsnn.models.GSNN import GSNN
from gsnn_lib.train.scGSNNTrainer import scGSNNTrainer

from geomloss import SamplesLoss    
from gsnn.optim.EarlyStopper import EarlyStopper
from gsnn_lib.train.EnergyDistanceLoss import EnergyDistanceLoss

import warnings
warnings.filterwarnings("ignore")

# TODO: EVAL delta r2 

def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../../proc/sciplex3/',
                        help="path to data directory")

    parser.add_argument("--out", type=str, default='../../output/sciplex3/gsnn/',
                        help="path to output directory")
    
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training iterations")
    
    parser.add_argument("--channels", type=int, default=5,
                        help="number of channels for each function node")
    
    parser.add_argument("--layers", type=int, default=10,
                        help="number of layers of message passing")
    
    parser.add_argument("--dropout", type=float, default=0.,
                        help="")
    
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate for the transport optimization")
        
    parser.add_argument("--save_every", type=int, default=10,
                        help="saves model results and weights every X epochs")
    
    parser.add_argument("--batch_size", type=int, default=64,
                        help="training batch size")
    
    parser.add_argument("--norm", type=str, default='layer',
                        help="GSNN normalization method to use [layer, batch, softmax, none]")
    
    parser.add_argument("--wd", type=float, default=0.,
                        help="weight decay")
    
    parser.add_argument("--blur", type=float, default=0.05,
                        help="the sinkhorn blur parameter")
    
    parser.add_argument("--scaling", type=float, default=0.9,
                        help="the sinkhorn scaling parameter [0,1]; higher is more accurate but slower")
    
    parser.add_argument("--reach", type=float, default=None,
                        help="the reach parameter for the sinkhorn distance; beneficial in noisy settings with outliers ")
    
    parser.add_argument("--p", type=int, default=2,
                        help="the p-norm for the sinkhorn distance; 1 is L1, 2 is L2")
    
    parser.add_argument("--checkpoint", action='store_true',
                        help="checkpoint GSNN layer gradients; will reduce memory but increase computation time")
    
    parser.add_argument("--debias", action='store_true',
                        help="debias argmuent to the sinkhorn distance")
    
    parser.add_argument("--ignore_cuda", action='store_true',
                        help="whether to ignore available cuda GPU")
    
    parser.add_argument("--patience", type=int, default=5,
                        help="epoch patience for early stopping")
    
    parser.add_argument("--min_delta", type=float, default=0.001,
                        help="minimum improvement for early stopping")
    
    parser.add_argument("--sched", type=str, default='none',
                        help="lr scheduler [onecycle, cosine, none]")
    
    parser.add_argument("--loss", type=str, default='sinkhorn',
                        help="loss function to use [sinkhorn, energy]")
    
    parser.add_argument("--optim", type=str, default='adam',
                        help="optimization algorithm to use [adam, adan, rmsprop, sgd]")
    
    args = parser.parse_args()

    return args


if __name__ == '__main__': 
    args = get_args()
    print(args, "\n")

    # Make output directory
    uid = str(uuid.uuid4())
    out_dir_ = os.path.join(args.out, uid)
    os.makedirs(out_dir_, exist_ok=True)
    print(f"UID: {uid}\nOutput directory: {out_dir_}\n")

    # Select device
    if torch.cuda.is_available() and not args.ignore_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    print('Using device:', device)

    # Load data
    data = torch.load(f'{args.data}/data.pt', weights_only=False)

    split_paths = np.sort([x for x in os.listdir(f'{args.data}/partitions/') if os.path.isdir(f'{args.data}/partitions/{x}')])
    print(split_paths)

    for split_path in split_paths: 

        out_dir = os.path.join(out_dir_, split_path)
        os.makedirs(out_dir, exist_ok=True)
        print(f"Split: {split_path} --> Output directory: {out_dir}\n")

        split_dict = torch.load(f'{args.data}/partitions/{split_path}/split_dict.pt')

        # Instantiate scSampler
        train_loader = scSampler(root=args.data, 
                                pert_ids=split_dict['pert']['train'], 
                                ctrl_ids=split_dict['ctrl']['train'], 
                                batch_size=args.batch_size,
                                ret_all_targets=True,
                                shuffle=True)
        
        val_loader = scSampler(root=args.data,
                                pert_ids=split_dict['pert']['val'],
                                ctrl_ids=split_dict['ctrl']['val'],
                                batch_size=args.batch_size,
                                ret_all_targets=True,
                                shuffle=False)
        
        test_loader = scSampler(root=args.data,
                                pert_ids=split_dict['pert']['test'],
                                ctrl_ids=split_dict['ctrl']['test'],
                                batch_size=args.batch_size,
                                ret_all_targets=True,
                                shuffle=False)

        # Build model
        model = GSNN(
            edge_index_dict=data.edge_index_dict,
            node_names_dict=data.node_names_dict,
            channels=args.channels,
            layers=args.layers,
            dropout=args.dropout,
            nonlin=torch.nn.ELU,  # example, can change
            bias=True,
            share_layers=False,
            add_function_self_edges=True,
            norm=args.norm,
            checkpoint=args.checkpoint
            # Additional arguments as needed
        ).to(device)

        # Define optimizer and criterion
        optim = utils.get_optim(args.optim)(model.parameters(), lr=args.lr, weight_decay=args.wd)

        if args.loss == 'sinkhorn':
            crit = SamplesLoss('sinkhorn', p=args.p, blur=args.blur, reach=args.reach, debias=args.debias, scaling=args.scaling)
        elif args.loss == 'energy':
            crit = EnergyDistanceLoss()
        else:
            raise ValueError(f'Unrecognized loss function: {args.loss}. Must be one of [sinkhorn, energy]')

        # Optionally define a scheduler, early_stopper, or logger
        scheduler = utils.get_scheduler(optim, args, train_loader)
        logger = utils.TBLogger(out_dir + '/tb/')
        stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)

        # Instantiate our scGSNNTrainer
        trainer = scGSNNTrainer(
            model=model,
            optimizer=optim,
            criterion=crit,
            device=device,
            logger=logger,
            scheduler=scheduler,
            early_stopper=stopper
        )

        # Train
        trainer.train(train_loader, val_loader, epochs=args.epochs, metric_key='neg_loss')

        # Test
        #test_metrics, y_test, yhat_test = trainer.test(test_loader)

        # load and predict ALL test data
        print('predict...')
        test_res_dict = {}
        for cond_idx in range(len(test_loader)):
            print(f'\tcondition {cond_idx+1}/{len(test_loader)}', end='\r')
            with torch.no_grad():

                y, cell_line, pert_id, dose_um = test_loader.sample_targets(cond_idx=cond_idx, ret_all=True)

                # NOTE: there will be a different number of inputs than targets 
                x, y0, cell_line, pert_id, dose_um = test_loader.sample_inputs(cond_idx=cond_idx, ret_all=True)

                yhat = [] 
                for ixs in torch.split(torch.arange(x.size(0)), args.batch_size): 
                    with torch.no_grad(): 
                        yhat.append( model(x[ixs].to(device)).detach().cpu() + y0[ixs] )
                yhat = torch.cat(yhat, dim=0)

                test_res_dict[cond_idx] = {'y': y.detach().cpu().numpy(), 
                                        'y0': y0.detach().cpu().numpy(), 
                                        'yhat': yhat.detach().cpu().numpy(), 
                                        'meta': {'cell_line': cell_line, 'pert_id': pert_id, 'dose_um': dose_um}}
                
        # Save test results
        torch.save(test_res_dict, os.path.join(out_dir, 'test_res_dict.pt'))

        #print("\nTest metrics:", test_metrics)

        # Save final model
        torch.save(model, os.path.join(out_dir, 'best_model.pt'))
        # Optionally save other results
        #np.save(os.path.join(out_dir, 'y_test.npy'), y_test)
        #np.save(os.path.join(out_dir, 'yhat_test.npy'), yhat_test)