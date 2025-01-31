import torch
import argparse
import uuid
import os
import time
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import pandas as pd

import sys
from gsnn.models.GSNN import GSNN
from gsnn_lib.data.LincsDataset import LincsDataset
from gsnn.models import utils
from gsnn_lib.proc.lincs.utils import get_x_drug_conc           # required to unpickle data
from gsnn.optim.EarlyStopper import EarlyStopper
from gsnn_lib.eval.eval import agg_fold_metrics, agg_fold_predictions, load_y, grouped_eval

from gsnn_lib.train.GSNNTrainer import GSNNTrainer  # Import the GSNNTrainer

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="You are using `torch.load` with `weights_only=False`.*"
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../../proc/lincs/',
                        help="path to data directory")
    parser.add_argument("--fold_dir", type=str, default='/partitions/',
                        help="relative path (from data) to partition splits information (dict .pt file)")
    parser.add_argument("--out", type=str, default='../../output/lincs/GSNN/',
                        help="path to output directory")
    parser.add_argument("--batch", type=int, default=25,
                        help="training batch size")
    parser.add_argument("--workers", type=int, default=12,
                        help="number of workers for dataloaders")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--randomize", action='store_true',
                        help="whether to randomize the structural graph")
    parser.add_argument("--ignore_cuda", action='store_true',
                        help="whether to ignore available cuda GPU")
    parser.add_argument("--channels", type=int, default=3,
                        help="number of channels for each function node")
    parser.add_argument("--layers", type=int, default=10,
                        help="number of layers of message passing")
    parser.add_argument("--dropout", type=float, default=0.,
                        help="")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--wd", type=float, default=0.,
                        help="weight decay")
    parser.add_argument("--nonlin", type=str, default='elu',
                        help="non-linearity function to use [relu, elu, mish, softplus, tanh, gelu]")
    parser.add_argument("--optim", type=str, default='adam',
                        help="optimization algorithm to use [adam, sgd, rmsprop]")
    parser.add_argument("--crit", type=str, default='mse',
                        help="loss function (criteria) to use [mse, huber]")
    parser.add_argument("--sched", type=str, default='none',
                        help="lr scheduler [onecycle, cosine, none]")
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="gradient clipping by norm")
    parser.add_argument("--no_bias", action='store_true',
                        help="whether to include a bias term in the function node neural networks.")
    parser.add_argument("--share_layers", action='store_true',
                        help="whether to share function node parameters across layers.")
    parser.add_argument("--checkpoint", action='store_true',
                        help="whether to use layer-wise gradient checkpointing")
    parser.add_argument("--add_function_self_edges", action='store_true',
                        help="Whether to add self-edges to function nodes.")
    parser.add_argument("--norm", type=str, default='layer',
                        help="normalization method to use [layer, none]")
    parser.add_argument("--init", type=str, default='kaiming',
                        help="weight initialization strategy: 'xavier', 'kaiming', 'lecun', 'normal'")
    parser.add_argument("--edge_channels", type=int, default=1,
                        help="number of duplicate edges to add additional edge latent channels")
    parser.add_argument("--compile", action='store_true',
                        help="use torch.compile on the model")
    parser.add_argument("--metric", type=str, default='r2',
                        help="metric to use for early stopping and best model [r2, mse]")
    parser.add_argument("--patience", type=int, default=10,
                        help="epoch patience for early stopping")
    parser.add_argument("--min_delta", type=float, default=0.001,
                        help="minimum improvement for early stopping")
    parser.add_argument("--prune_every", type=int, default=None,
                        help="prune network weights every n epochs")
    parser.add_argument("--prune_threshold", type=float, default=1e-2,
                        help="pruning threshold; weights with absolute value less than this will be removed during training")
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

    torch.save(data, out_dir + '/Data.pt')

    model = GSNN(
        edge_index_dict             = data.edge_index_dict,
        node_names_dict             = data.node_names_dict,
        channels                    = args.channels,
        layers                      = args.layers,
        dropout                     = args.dropout,
        nonlin                      = utils.get_activation(args.nonlin),
        bias                        = not args.no_bias,
        share_layers                = args.share_layers,
        add_function_self_edges     = args.add_function_self_edges,
        norm                        = args.norm,
        init                        = args.init,
        checkpoint                  = args.checkpoint,
        edge_channels               = args.edge_channels,
    ).to(device)

    if args.compile:
        print('compiling model...')
        model = torch.compile(model)

    n_params = sum([p.numel() for p in model.parameters()])
    args.n_params = n_params
    print('# params', n_params)

    optim = utils.get_optim(args.optim)
    optim_params = {'lr':args.lr, 'weight_decay':args.wd}
    crit  = utils.get_crit(args.crit)()
    scheduler = None
    logger = utils.TBLogger(out_dir + '/tb/')
    stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)

    # Create a GSNNTrainer instance
    trainer = GSNNTrainer(
        model=model,
        optimizer=optim,
        optim_params=optim_params,
        criterion=crit,
        device=device,
        logger=logger,
        scheduler=scheduler,
        early_stopper=stopper,
        prune_every=args.prune_every, 
        prune_threshold=args.prune_threshold
    )

    # Train the model
    trainer.train(train_loader, val_loader, epochs=args.epochs, metric_key=args.metric)

    # Save the best model
    torch.save(model, out_dir + f'/best_model.pt')

    # log hyperparameters and compute final metrics
    time_elapsed = time.time() - time0
    metric_dict, yhat_test, sig_ids_test = logger.add_hparam_results(args=args,
                                                                    model=model,
                                                                    data=data,
                                                                    device=device,
                                                                    test_loader=test_loader,
                                                                    val_loader=val_loader,
                                                                    siginfo=condinfo,
                                                                    time_elapsed=time_elapsed,
                                                                    epoch=trainer.current_epoch)

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
    args.model = 'gsnn'

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
    # NOTE: all folds will have the same randomized graph structure
    if args.randomize: data.edge_index_dict = utils.randomize(data)

    folds = np.sort([x for x in os.listdir(f'{args.data}/{args.fold_dir}') if 'lincs' in x])
    print('# folds:', len(folds))

    for fold in folds:
        torch.cuda.empty_cache()
        print('########################################################################################')
        print(f'running fold: {fold}')
        train_fold(args, fold, out_dir, data, condinfo, device)
        print('########################################################################################')

    # aggregate all metrics
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
