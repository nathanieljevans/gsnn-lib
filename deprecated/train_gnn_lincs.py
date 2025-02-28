import torch 
import argparse 
import uuid
import os 
import time
import numpy as np 
from sklearn.metrics import r2_score
import torch_geometric as pyg 
import pandas as pd 

import warnings
warnings.filterwarnings("ignore")
'''
UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
'''

import sys 
from gsnn_lib.data.pygLincsDataset import pygLincsDataset
from gsnn.models import utils 

def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default='../processed_data/',
                        help="path to data directory")
    
    parser.add_argument("--fold", type=str, default='../processed_data/fold/',
                        help="path to data fold directory; must contain data splits - see `create_data_splits.py`")
    
    parser.add_argument("--siginfo", type=str, default='../../data/',
                        help="path to siginfo directory")
    
    parser.add_argument("--out", type=str, default='../output/',
                        help="path to output directory")
    
    parser.add_argument("--gnn", type=str, default='GCN',
                        help="GNN archtecture to use, supports: GCN, GAT, GIN")
    
    parser.add_argument("--batch", type=int, default=100,
                        help="training batch size")
    
    parser.add_argument("--workers", type=int, default=12,
                        help="number of workers for dataloaders")
    
    parser.add_argument("--epochs", type=int, default=50,
                        help="number of training epochs")
    
    parser.add_argument("--randomize", action='store_true',
                        help="whether to randomize the structural graph")
    
    parser.add_argument("--ignore_cuda", action='store_true',
                        help="whether to ignore available cuda GPU")
    
    parser.add_argument("--channels", type=int, default=32,
                        help="number of channels for each function node")
    
    parser.add_argument("--layers", type=int, default=10,
                        help="number of layers of message passing")
    
    parser.add_argument("--dropout", type=float, default=0.,
                        help="")
    
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    
    parser.add_argument("--wd", type=float, default=0.,
                        help="weight decay")
    
    parser.add_argument("--optim", type=str, default='adam',
                        help="optimization algorithm to use [adam, sgd, rmsprop]")
    
    parser.add_argument("--crit", type=str, default='mse',
                        help="loss function (criteria) to use [mse, huber]")
    
    parser.add_argument("--norm", type=str, default='batch',
                        help="normalization function to use [none, batch]")
    
    parser.add_argument("--jk", type=str, default='cat',
                        help="normalization function to use [cat, max, lstm]")
    
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="gradient clipping by norm")
    
    parser.add_argument("--sched", type=str, default='none',
                        help="lr scheduler [onecycle, cosine, none]")
    
    parser.add_argument("--nonlin", type=str, default='elu',
                        help="non-linearity function to use")
    
    parser.add_argument("--save_every", type=int, default=10,
                        help="saves model results and weights every X epochs")
    
    return parser.parse_args()
    


if __name__ == '__main__': 

    time0 = time.time() 

    # get args 
    args = get_args()
    args.model = 'gnn'
    args.cell_agnostic = False

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
        for i in range(torch.cuda.device_count()): print(f'cuda device {i}: {torch.cuda.get_device_properties(i).name}')
    else: 
        device = 'cpu'
    args.device = device
    print('using device:', device)

    data = torch.load(f'{args.data}/Data.pt')

    train_ids = np.load(f'{args.fold}/lincs_train_obs.npy', allow_pickle=True)
    train_dataset = pygLincsDataset(root=f'{args.data}', sig_ids=train_ids, data=data)
    train_loader = pyg.loader.DataLoader(train_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=True)

    test_ids = np.load(f'{args.fold}/lincs_test_obs.npy', allow_pickle=True)
    test_dataset = pygLincsDataset(root=f'{args.data}', sig_ids=test_ids, data=data)
    test_loader = pyg.loader.DataLoader(test_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=False)

    val_ids = np.load(f'{args.fold}/lincs_val_obs.npy', allow_pickle=True)
    val_dataset = pygLincsDataset(root=f'{args.data}', sig_ids=val_ids, data=data)
    val_loader = pyg.loader.DataLoader(val_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=False)

    if args.randomize: 
        data.edge_index = utils.randomize(data)

    torch.save(data, out_dir + '/Data.pt')

    if args.gnn == 'GAT': 
        GNN = pyg.nn.models.GAT
    elif args.gnn == 'GCN': 
        GNN = pyg.nn.models.GAT
    elif args.gnn == 'GIN': 
        GNN = pyg.nn.models.GIN
    else: 
        raise ValueError(f'unrecognized `gnn` value. Expected one of ["GAT" "GCN" "GIN"] but got: {args.gnn}')

    model = GNN(in_channels=1, 
                hidden_channels=args.channels, 
                num_layers=args.layers,
                out_channels=1, 
                dropout=args.dropout, 
                act=args.nonlin,
                act_first=False,
                act_kwargs=None,
                norm=args.norm,
                norm_kwargs=None,
                jk=args.jk).to(device)
    
    n_params = sum([p.numel() for p in model.parameters()])
    args.n_params = n_params
    print('# params', n_params)

    optim = utils.get_optim(args.optim)(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = utils.get_crit(args.crit)()
    scheduler = utils.get_scheduler(optim, args, train_loader)
    logger = utils.TBLogger(out_dir + '/tb/')

    siginfo = pd.read_csv(f'{args.siginfo}/siginfo_beta.txt', sep='\t', low_memory=False)

    for epoch in range(1, args.epochs+1):
        big_tic = time.time()
        model = model.train()
        losses = []
        for i, batch in enumerate(train_loader): 

            if len(batch.sig_id) == 1: 
                # BUG: if batch only has 1 obs it fails
                continue 

            tic = time.time()
            optim.zero_grad() 

            yhat = model(edge_index=batch.edge_index.to(device), x=batch.x.to(device))

            #  select output nodes
            yhat = yhat[batch.output_node_mask]
            y = batch.y.to(device)[batch.output_node_mask]

            loss = crit(yhat, y)
            loss.backward()
            if args.clip_grad is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optim.step()
            if scheduler is not None: scheduler.step()

            with torch.no_grad(): 

                B = len(batch.sig_id)

                yhat = yhat.view(B, -1).detach().cpu().numpy() 
                y = y.view(B, -1).detach().cpu().numpy() 
                r2 = r2_score(y, yhat, multioutput='variance_weighted')
                r_flat = np.corrcoef(y.ravel(), yhat.ravel())[0,1]
                losses.append(loss.item())

                print(f'epoch: {epoch} || batch: {i}/{len(train_loader)} || loss: {loss.item():.3f} || r2: {r2:.3f} || r (flat): {r_flat:.2f} || elapsed: {(time.time() - tic):.2f} s' , end='\r')
        
        loss_train = np.mean(losses)

        y,yhat,sig_ids = utils.predict_gnn(test_loader, model, data, device)
        r_cell, r_drug, r_dose = utils._get_regressed_metrics(y, yhat, sig_ids, siginfo)
        r2_val = r2_score(y, yhat, multioutput='variance_weighted')
        r_flat_val = np.corrcoef(y.ravel(), yhat.ravel())[0,1]

        logger.log(epoch, loss_train, r2_val, r_flat_val)
        
        if (epoch % args.save_every == 0): 

            time_elapsed = time.time() - time0
            # add test results + hparams
            logger.add_hparam_results(args=args, 
                                    model=model, 
                                    data=data, 
                                    device=device, 
                                    test_loader=test_loader, 
                                    val_loader=val_loader, 
                                    siginfo=siginfo,
                                    time_elapsed=time_elapsed,
                                    epoch=epoch)

            torch.save(model, out_dir + f'/model-{epoch}.pt')

        print(f'Epoch: {epoch} || loss (train): {loss_train:.3f} || r2 (val): {r2_val:.2f} || r flat (val): {r_flat_val:.2f} || r cell: {r_cell:.2f} || r drug: {r_drug:.2f} || elapsed: {(time.time() - big_tic)/60:.2f} min')

