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
from gsnn.data.LincsDataset import LincsDataset
from gsnn.models import utils 
from gsnn.proc.utils import get_x_drug_conc           # required to unpickle data 

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
    
    parser.add_argument("--batch", type=int, default=25,
                        help="training batch size")
    
    parser.add_argument("--workers", type=int, default=12,
                        help="number of workers for dataloaders")
    
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    
    parser.add_argument("--randomize", action='store_true',
                        help="whether to randomize the structural graph")
    
    parser.add_argument("--no_residual", action='store_true',
                        help="disable residual connections")
    
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
    
    parser.add_argument("--nonlin", type=str, default='gelu',
                        help="non-linearity function to use [relu, elu, mish, softplus, tanh, gelu]")
    
    parser.add_argument("--optim", type=str, default='adam',
                        help="optimization algorithm to use [adam, sgd, rmsprop]")
    
    parser.add_argument("--crit", type=str, default='mse',
                        help="loss function (criteria) to use [mse, huber]")
    
    parser.add_argument("--sched", type=str, default='none',
                        help="lr scheduler [onecycle, cosine, none]")
    
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="gradient clipping by norm")
    
    parser.add_argument("--save_every", type=int, default=25,
                        help="saves model results and weights every X epochs")
    
    parser.add_argument("--report_every", type=int, default=5,
                        help="report validation set performances every X epochs")
    
    parser.add_argument("--no_bias", action='store_true',
                        help="whether to include a bias term in the function node neural networks.")
    
    parser.add_argument("--share_layers", action='store_true',
                        help="whether to share function node parameters across layers.")
    
    parser.add_argument("--scale_channels_by_degree", action='store_true',
                        help="if False, all function nodes will have `channels` hidden units, otherwise, the number of hidden channels of each function node will depend on the function node degree (e.g., nodes with more inputs/output will have more hidden channels.).")
    
    parser.add_argument("--two_layer_conv", action='store_true',
                        help="Number of layers in each function node neural network. If True, then two layers, otherwise 1.")
    
    parser.add_argument("--checkpoint", action='store_true',
                        help="whether to use layer-wise gradient checkpointing; can reduce memory usage significantly.")
    
    parser.add_argument("--add_function_self_edges", action='store_true',
                        help="Whether to add self-edges to function nodes.")
    
    parser.add_argument("--norm", type=str, default='layer',
                        help="normalization method to use [layer, none]")
    
    parser.add_argument("--init", type=str, default='kaiming',
                        help="weight initialization strategy: 'xavier'-or-'glorot', 'kaiming'-or-'he', 'lecun', normal'")
    
    parser.add_argument("--edge_channels", type=int, default=1,
                        help="number of duplicate edges to make additional edge latent channels; 1 will not duplicate")
    
    parser.add_argument("--dropout_type", type=str, default='channel',
                        help="dropout style to use [node, channel, edge]")
    
    parser.add_argument("--compile", action='store_true',
                        help="use torch.compile on the model; may improve training speeds")
    
    args = parser.parse_args()

    return args

def step(model, x, y, crit, args): 
    '''run one gradient descent step'''
    model.train()
    optim.zero_grad()
    yhat = model(x)
    loss = crit(yhat, y)

    loss.backward()
    if args.clip_grad is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optim.step()

    return loss, yhat


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
        for i in range(torch.cuda.device_count()): print(f'cuda device {i}: {torch.cuda.get_device_properties(i).name}')
    else: 
        device = 'cpu'
    args.device = device
    print('using device:', device)

    siginfo = pd.read_csv(f'{args.siginfo}/siginfo_beta.txt', sep='\t', low_memory=False)

    data = torch.load(f'{args.data}/data.pt')

    train_ids = np.load(f'{args.fold}/lincs_train_obs.npy', allow_pickle=True)
    train_dataset = LincsDataset(root=f'{args.data}', sig_ids=train_ids, data=data, siginfo=siginfo)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=True, persistent_workers=True)

    test_ids = np.load(f'{args.fold}/lincs_test_obs.npy', allow_pickle=True)
    test_dataset = LincsDataset(root=f'{args.data}', sig_ids=test_ids, data=data, siginfo=siginfo)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=False)

    val_ids = np.load(f'{args.fold}/lincs_val_obs.npy', allow_pickle=True)
    val_dataset = LincsDataset(root=f'{args.data}', sig_ids=val_ids, data=data, siginfo=siginfo)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=False)

    if args.randomize: data.edge_index = utils.randomize(data)

    torch.save(data, out_dir + '/Data.pt')

    model = GSNN(edge_index_dict                 = data.edge_index_dict, 
                 node_names_dict                 = data.node_names_dict,
                channels                        = args.channels, 
                layers                          = args.layers, 
                dropout                         = args.dropout,
                residual                        = not args.no_residual,
                nonlin                          = utils.get_activation(args.nonlin),
                bias                            = not args.no_bias,
                share_layers                    = args.share_layers,
                fix_hidden_channels             = not args.scale_channels_by_degree,
                two_layer_conv                  = args.two_layer_conv, 
                add_function_self_edges         = args.add_function_self_edges,
                norm                            = args.norm,
                init                            = args.init,
                checkpoint                      = args.checkpoint,
                edge_channels                   = args.edge_channels,
                dropout_type                    = args.dropout_type).to(device)
    
    if args.compile: 
        print('compiling model...')
        model = torch.compile(model)
        
    n_params = sum([p.numel() for p in model.parameters()])
    args.n_params = n_params
    print('# params', n_params)

    optim       = utils.get_optim(args.optim)(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit        = utils.get_crit(args.crit)()
    scheduler   = utils.get_scheduler(optim, args, train_loader)
    logger      = utils.TBLogger(out_dir + '/tb/')

    # get indices of all non-drug input nodes 
    # necessary for cell agnostic and flag 
    #omic_input_mask = (data.input_node_mask & torch.tensor(['DRUG_' not in x for x in data.node_names], dtype=torch.bool)).to(device)
    #omic_input_idxs = omic_input_mask.nonzero(as_tuple=True)[0].to(device)
    #omic_input_mask = omic_input_mask.view(1,-1,1)

    for epoch in range(1, args.epochs+1):
        big_tic = time.time()
        model = model.train()
        losses = []
        for i,(x, y, sig_id) in enumerate(train_loader): 
            tic = time.time()

            if len(sig_id) == 1: continue # BUG workaround: if batch only has 1 obs it fails

            loss, yhat = step(model, 
                                x.to(device), 
                                y.to(device), 
                                crit, 
                                args)

            if scheduler is not None: scheduler.step()

            with torch.no_grad(): 

                yhat    = yhat.detach().cpu().numpy() 
                y       = y.detach().cpu().numpy() 
                
                r2      = r2_score(y, yhat, multioutput='variance_weighted')
                r_flat  = np.corrcoef(y.ravel(), yhat.ravel())[0,1]
                
                losses.append(loss.item())

                print(f'epoch: {epoch} || batch: {i}/{len(train_loader)} || loss: {loss.item():.3f} || r2: {r2:.3f} || r (flat): {r_flat:.2f} || elapsed: {(time.time() - tic):.2f} s' , end='\r')
        
        if (epoch % args.report_every == 0): 
            loss_train = np.mean(losses)

            y,yhat,sig_ids                          = utils.predict_gsnn(val_loader, model, device)
            #r_cell, r_drug, r_dose                  = utils._get_regressed_metrics(y, yhat, sig_ids, siginfo)
            r2_val                                  = r2_score(y, yhat, multioutput='variance_weighted')
            r_flat_val                              = np.corrcoef(y.ravel(), yhat.ravel())[0,1]

            logger.log(epoch, loss_train, r2_val, r_flat_val)

            print(f'Epoch: {epoch} || loss (train): {loss_train:.3f} || r2 (val): {r2_val:.2f} || r flat (val): {r_flat_val:.2f} || elapsed: {(time.time() - big_tic)/60:.2f} min')
            #print(f'Epoch: {epoch} || loss (train): {loss_train:.3f} || r2 (val): {r2_val:.2f} || r flat (val): {r_flat_val:.2f} || r cell: {r_cell:.2f} || r drug: {r_drug:.2f} || elapsed: {(time.time() - big_tic)/60:.2f} min')

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