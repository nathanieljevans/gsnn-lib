import torch
import argparse
import uuid
import os
import time
import numpy as np
import pandas as pd
import torch_geometric as pyg
import sys
from gsnn_lib.data.pygLincsDataset import pygLincsDataset
from gsnn.models import utils
from gsnn_lib.train.GNNTrainer import GNNTrainer
from gsnn.optim.EarlyStopper import EarlyStopper
from gsnn_lib.eval.eval import agg_fold_metrics, agg_fold_predictions, load_y, grouped_eval

import warnings
warnings.filterwarnings("ignore")

# BUG: "received 0 items of ancdata" error (soln: https://stackoverflow.com/questions/71642653/how-to-resolve-the-error-runtimeerror-received-0-items-of-ancdata ; @KingLiu)
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../../proc/lincs/',
                        help="path to data directory")
    parser.add_argument("--fold_dir", type=str, default='/partitions/',
                        help="relative path (from data) to partition splits")
    parser.add_argument("--out", type=str, default='../../output/GNN/',
                        help="path to output directory")
    parser.add_argument("--gnn", type=str, default='GAT',
                        help="GNN architecture: [GAT, GIN]")
    parser.add_argument("--batch", type=int, default=50,
                        help="training batch size")
    parser.add_argument("--workers", type=int, default=6,
                        help="number of workers for dataloaders")
    parser.add_argument("--epochs", type=int, default=50,
                        help="number of training epochs")
    parser.add_argument("--randomize", action='store_true',
                        help="whether to randomize the structural graph")
    parser.add_argument("--ignore_cuda", action='store_true',
                        help="whether to ignore available cuda GPU")
    parser.add_argument("--channels", type=int, default=32,
                        help="hidden channels for GNN")
    parser.add_argument("--layers", type=int, default=10,
                        help="number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout rate")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--wd", type=float, default=0.,
                        help="weight decay")
    parser.add_argument("--optim", type=str, default='adam',
                        help="optimizer [adam, sgd, rmsprop]")
    parser.add_argument("--crit", type=str, default='mse',
                        help="loss function [mse, huber]")
    parser.add_argument("--norm", type=str, default='layer',
                        help="normalization: [none, batch, layer, pairnorm]")
    parser.add_argument("--jk", type=str, default='cat',
                        help="jumping knowledge style: [cat, max, lstm, none]")
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="gradient clipping")
    parser.add_argument("--sched", type=str, default='none',
                        help="lr scheduler [onecycle, cosine, none]")
    parser.add_argument("--nonlin", type=str, default='elu',
                        help="non-linearity function")
    parser.add_argument("--metric", type=str, default='r2',
                        help="metric for early stopping [r2, mse]")
    parser.add_argument("--patience", type=int, default=5,
                        help="early stopping patience")
    parser.add_argument("--min_delta", type=float, default=0.001,
                        help="minimum improvement for early stopping")

    args = parser.parse_args()
    return args

def build_gnn(args):
    # map user argument to actual GNN class
    if args.gnn == 'GAT':
        GNN = pyg.nn.models.GAT
        kwargs = {'heads': 1, 'add_self_loops': False}
    elif args.gnn == 'GIN':
        GNN = pyg.nn.models.GIN
        kwargs = {}
    else:
        raise ValueError(f'Unrecognized gnn: {args.gnn}. Must be one of [GAT, SAGE, GIN]')

    model = GNN(
        in_channels=1,  # adjust if needed
        hidden_channels=args.channels,
        num_layers=args.layers,
        out_channels=1,
        dropout=args.dropout,
        act=args.nonlin,
        act_first=False,
        act_kwargs=None,
        norm=args.norm if args.norm != 'none' else None,
        norm_kwargs=None,
        jk=args.jk if args.jk != 'none' else None,
        **kwargs
    )

    #NOTE: adding dummy inp->inp and out->out edge types even tho we don't have them (needed for hetero conversion)
    metadata = (['input', 'function', 'output'], 
                [('input','to','function'), ('input', 'to', 'input'), ('function','to','function'), ('function','to','output'), ('output','to','output')])
    model = pyg.nn.to_hetero(model, metadata, aggr='sum')

    return model

def train_fold(args, fold, out_dir, data, condinfo, device, time0):

    out_dir = f'{out_dir}/{fold.split(".")[0]}'
    os.makedirs(out_dir)

    split_dict = torch.load(f'{args.data}/{args.fold_dir}/{fold}')

    train_ids = split_dict['train_obs']
    val_ids = split_dict['val_obs']
    test_ids = split_dict['test_obs']

    train_dataset = pygLincsDataset(root=f'{args.data}', cond_ids=train_ids, data=data, condinfo=condinfo)
    val_dataset = pygLincsDataset(root=f'{args.data}', cond_ids=val_ids, data=data, condinfo=condinfo)
    test_dataset = pygLincsDataset(root=f'{args.data}', cond_ids=test_ids, data=data, condinfo=condinfo)

    train_loader = pyg.loader.DataLoader(train_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=True, persistent_workers=True)
    val_loader = pyg.loader.DataLoader(val_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=False)
    test_loader = pyg.loader.DataLoader(test_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=False)

    print('train:', len(train_dataset), 'test:', len(test_dataset), 'val:', len(val_dataset))

    # Save data snapshot
    torch.save(data, out_dir + '/Data.pt')

    # Build model
    model = build_gnn(args).to(device)

    n_params = sum([p.numel() for p in model.parameters()])
    args.n_params = n_params
    print('# params', n_params)

    optim = utils.get_optim(args.optim)(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = utils.get_crit(args.crit)()
    scheduler = utils.get_scheduler(optim, args, train_loader)
    logger = utils.TBLogger(out_dir + '/tb/')
    stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)

    trainer = GNNTrainer(
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

    # Save best model
    torch.save(model, out_dir + f'/best_model.pt')

    time_elapsed = time.time() - time0

    # Add hparam results
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

    args = get_args()
    args.model = 'gnn'
    args.cell_agnostic = False

    print()
    print(args)
    print()

    os.makedirs(args.out, exist_ok=True)

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
    data = torch.load(f'{args.data}/data.pt')

    # optionally randomize graph
    if args.randomize:
        data.edge_index = utils.randomize(data)

    # Run across folds
    folds = np.sort([x for x in os.listdir(f'{args.data}/{args.fold_dir}') if 'lincs' in x])
    print('# folds:', len(folds))

    for fold in folds:
        torch.cuda.empty_cache()
        print('########################################################################################')
        print(f'running fold: {fold}')
        train_fold(args, fold, out_dir, data, condinfo, device, time0)
        print('########################################################################################')

    # aggregate all metrics
    torch.cuda.empty_cache()
    path = out_dir
    proc = args.data

    avg_metrics = agg_fold_metrics(path)
    preds, sig_ids = agg_fold_predictions(path)
    y = load_y(proc, sig_ids)
    drug_metrics, cell_metrics, gene_metrics = grouped_eval(y, preds, sig_ids, data, condinfo)

    torch.save(avg_metrics, f'{path}/avg_metrics.pt')

    drug_metrics.to_csv(f'{path}/test_drug_metrics.csv', index=False)
    cell_metrics.to_csv(f'{path}/test_cell_metrics.csv', index=False)
    gene_metrics.to_csv(f'{path}/test_gene_metrics.csv', index=False)
