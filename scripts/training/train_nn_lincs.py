import torch
import argparse
import uuid
import os
import time
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

import sys
from gsnn.models.NN import NN
from gsnn_lib.data.LincsDataset import LincsDataset
from gsnn.models import utils
from gsnn.optim.EarlyStopper import EarlyStopper
from gsnn_lib.train.NNTrainer import NNTrainer
from gsnn_lib.eval.eval import agg_fold_metrics, agg_fold_predictions, load_y, grouped_eval

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="You are using `torch.load` with `weights_only=False`.*"
)


##############################################################################################################################
##############################################################################################################################

class AE(torch.nn.Module): 

    def __init__(self, data, hidden_channels, latent_dim, out_channels, layers=2, dropout=0, 
                        nonlin=torch.nn.ELU, out=None, norm=torch.nn.BatchNorm1d): 
        '''
        
        Args: 
            in_channels             int                 number of input channels 
            hidden_channels         int                 number of hidden channels per layer 
            out_channels            int                 number of output channels 
            layers                  int                 number of hidden layers 
            dropout                 float               dropout regularization probability 
            nonlin                  pytorch.module      non-linear activation function 
            out                     pytorch.module      output transformation to be applied 
            norm                    pytorch.module      normalization method to use 
        '''
        super().__init__()
        
        n_drug_features = len([x for x in data.node_names_dict['input'] if 'DRUG__' in x])
        self.register_buffer('drug_input_ixs', torch.tensor([i for i,x in enumerate(data.node_names_dict['input']) if 'DRUG__' in x], dtype=torch.long)) 
        self.drug_enc = NN(in_channels=n_drug_features, hidden_channels=hidden_channels, out_channels=latent_dim,
                                layers=layers, dropout=dropout, nonlin=nonlin, out=out, norm=norm)
        
        n_cell_features = len([x for x in data.node_names_dict['input'] if 'DRUG__' not in x])
        self.register_buffer('cell_input_ixs', torch.tensor([i for i,x in enumerate(data.node_names_dict['input']) if 'DRUG__' not in x], dtype=torch.long))
        self.cell_enc = NN(in_channels=n_cell_features, hidden_channels=hidden_channels, out_channels=latent_dim,
                                layers=layers, dropout=dropout, nonlin=nonlin, out=out, norm=norm)  
        
        self.dec = NN(in_channels=latent_dim, hidden_channels=hidden_channels, out_channels=out_channels,
                        layers=layers, dropout=dropout, nonlin=nonlin, out=out, norm=norm) 
        

    def forward(self, x): 

        x_drug = x[:, self.drug_input_ixs]
        x_cell = x[:, self.cell_input_ixs]

        z_drug = self.drug_enc(x_drug)
        z_cell = self.cell_enc(x_cell)

        z = z_drug + z_cell

        xhat = self.dec(z)

        return xhat


##############################################################################################################################
##############################################################################################################################


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../proc/lincs/',
                        help="path to data directory")
    parser.add_argument("--fold_dir", type=str, default='/partitions/',
                        help="relative path (from data) to partition splits information (dict .pt file)")
    parser.add_argument("--out", type=str, default='../output/',
                        help="path to output directory")

    parser.add_argument("--arch", type=str, default='nn',
                        help="model architecture to use [nn, ae]")

    parser.add_argument("--batch", type=int, default=512,
                        help="training batch size")
    parser.add_argument("--workers", type=int, default=12,
                        help="number of workers for dataloaders")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--randomize", action='store_true',
                        help="whether to randomize the structural graph")
    parser.add_argument("--ignore_cuda", action='store_true',
                        help="whether to ignore available cuda GPU")

    # NN-specific arguments (similar structure as GSNN)
    parser.add_argument("--channels", type=int, default=124,
                        help="hidden size for NN model")
    parser.add_argument("--layers", type=int, default=2,
                        help="number of layers for NN model")
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout rate for NN")
    parser.add_argument("--nonlin", type=str, default='elu',
                        help="non-linearity function to use [relu, elu, gelu, etc.]")
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="latent dimension for autoencoder (not relevant to NN)")

    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--wd", type=float, default=0.,
                        help="weight decay")
    parser.add_argument("--optim", type=str, default='adam',
                        help="optimization algorithm to use [adam, sgd, rmsprop]")
    parser.add_argument("--crit", type=str, default='mse',
                        help="loss function (criteria) to use [mse, huber]")
    parser.add_argument("--sched", type=str, default='none',
                        help="lr scheduler [onecycle, cosine, none]")
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="gradient clipping by norm")

    parser.add_argument("--metric", type=str, default='r2',
                        help="metric to use for early stopping and best model [r2, mse]")
    parser.add_argument("--patience", type=int, default=5,
                        help="epoch patience for early stopping")
    parser.add_argument("--min_delta", type=float, default=0.001,
                        help="minimum improvement for early stopping")

    args = parser.parse_args()
    return args

def train_fold(args, fold, out_dir, data, condinfo, device):

    out_dir = f'{out_dir}/{fold.split(".")[0]}'
    os.makedirs(out_dir)

    split_dict = torch.load(f'{args.data}/{args.fold_dir}/{fold}')

    train_dataset = LincsDataset(root=f'{args.data}', cond_ids=split_dict['train_obs'], data=data, cond_meta=condinfo)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=True, persistent_workers=True)

    test_dataset = LincsDataset(root=f'{args.data}', cond_ids=split_dict['test_obs'], data=data, cond_meta=condinfo)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=False)

    val_dataset = LincsDataset(root=f'{args.data}', cond_ids=split_dict['val_obs'], data=data, cond_meta=condinfo)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=False)

    print('train:', len(train_dataset), 'test:', len(test_dataset), 'val:', len(val_dataset))

    # Save data snapshot
    torch.save(data, out_dir + '/Data.pt')

    # Initialize NN model
    if args.arch == 'nn': 
        model = NN(
            in_channels=len(data.node_names_dict['input']),
            hidden_channels=args.channels,
            out_channels=len(data.node_names_dict['output']),
            layers=args.layers,
            dropout=args.dropout,
            nonlin=utils.get_activation(args.nonlin)
        ).to(device)
    elif args.arch == 'ae':
        model = AE(
            data=data,
            hidden_channels=args.channels,
            latent_dim=args.latent_dim,
            out_channels=len(data.node_names_dict['output']),
            layers=args.layers,
            dropout=args.dropout,
            nonlin=utils.get_activation(args.nonlin)
        ).to(device)
    else: 
        raise ValueError(f'Unrecognized architecture: {args.arch}. Must be one of [nn, ae]')

    n_params = sum([p.numel() for p in model.parameters()])
    args.n_params = n_params
    print('# params', n_params)

    optim = utils.get_optim(args.optim)(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit  = utils.get_crit(args.crit)()
    scheduler = utils.get_scheduler(optim, args, train_loader)
    logger = utils.TBLogger(out_dir + '/tb/')
    stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)

    # Create a NNTrainer instance
    trainer = NNTrainer(
        model=model,
        optimizer=optim,
        criterion=crit,
        device=device,
        logger=logger,
        scheduler=scheduler,
        early_stopper=stopper
    )

    # Train the model
    trainer.train(train_loader, val_loader, epochs=args.epochs, metric_key=args.metric)

    # Save the best model
    torch.save(model, out_dir + f'/best_model.pt')

    # Evaluate and log hyperparameters
    time_elapsed = time.time() - time0
    metric_dict, yhat_test, sig_ids_test = logger.add_hparam_results(
        args=args,
        model=model,
        data=data,
        device=device,
        test_loader=test_loader,
        val_loader=val_loader,
        siginfo=condinfo,
        time_elapsed=time_elapsed,
        epoch=trainer.current_epoch
    )

    torch.save(metric_dict, f'{out_dir}/result_metric_dict.pt')
    torch.save(yhat_test, f'{out_dir}/test_predictions.pt')
    torch.save(sig_ids_test, f'{out_dir}/test_sig_ids.pt')

    print()
    print('best model metrics:')
    for k, v in metric_dict.items():
        print(f'\t{k}: {v}')

if __name__ == '__main__':

    time0 = time.time()

    # get args
    args = get_args()
    args.model = 'nn'
    args.randomize = False

    print()
    print(args)
    print()

    # create uuid
    uid = str(uuid.uuid4())
    args.uid = uid
    print('UID:', uid)
    out_dir = f'{args.out}/{uid}'
    if not os.path.exists(args.out):
        os.mkdir(args.out)
    os.mkdir(out_dir)

    with open(f'{out_dir}/args.log', 'w') as f:
        f.write(str(args))

    if torch.cuda.is_available() and not args.ignore_cuda:
        device = 'cuda'
        for i in range(torch.cuda.device_count()):
            print(f'cuda device {i}: {torch.cuda.get_device_properties(i).name}')
    else:
        device = 'cpu'
    args.device = device
    print('using device:', device)

    condinfo = pd.read_csv(f'{args.data}/conditions_meta.csv', sep=',')
    data = torch.load(f'{args.data}/data.pt', weights_only=False)

    # optionally randomize graph
    if args.randomize:
        data.edge_index = utils.randomize(data)

    # In the GSNN script, we load multiple folds and run train_fold on each.
    # For the NN script, let's emulate that behavior.
    folds = np.sort([x for x in os.listdir(f'{args.data}/{args.fold_dir}') if 'lincs' in x])
    print('# folds:', len(folds))

    for fold in folds:
        torch.cuda.empty_cache()
        print('########################################################################################')
        print(f'running fold: {fold}')
        train_fold(args, fold, out_dir, data, condinfo, device)
        print('########################################################################################')

    # aggregate all metrics like GSNN does
    torch.cuda.empty_cache()
    path = out_dir
    proc = args.data

    avg_metrics = agg_fold_metrics(path)
    preds, cond_ids = agg_fold_predictions(path)
    y = load_y(proc, cond_ids)
    drug_metrics, cell_metrics, gene_metrics = grouped_eval(y, preds, cond_ids, data, condinfo)

    torch.save(avg_metrics, f'{path}/avg_metrics.pt')

    drug_metrics.to_csv(f'{path}/test_drug_metrics.csv', index=False)
    cell_metrics.to_csv(f'{path}/test_cell_metrics.csv', index=False)
    gene_metrics.to_csv(f'{path}/test_gene_metrics.csv', index=False)
