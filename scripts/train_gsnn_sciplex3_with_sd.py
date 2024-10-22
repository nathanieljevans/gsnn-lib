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

from hnet.models.HyperNet import HyperNet
from hnet.train.hnet import init_hnet

import sys 
from gsnn.models.GSNN import GSNN
from gsnn.models import utils 
from gsnn_lib.proc.lincs.utils import get_x_drug_conc           # required to unpickle data 
from gsnn.models.NN import NN
from gsnn_lib.data.scSampler import scSampler

from geomloss import SamplesLoss

import warnings
warnings.filterwarnings("ignore")

import gc

def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../sc_data/',
                        help="path to data directory")

    parser.add_argument("--out", type=str, default='../sc_output/',
                        help="path to output directory")
    
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    
    parser.add_argument("--channels", type=int, default=5,
                        help="number of channels for each function node")
    
    parser.add_argument("--layers", type=int, default=10,
                        help="number of layers of message passing")
    
    parser.add_argument("--dropout", type=float, default=0.,
                        help="")
        
    parser.add_argument("--wd", type=float, default=1e-10,
                        help="weight decay")
    
    parser.add_argument("--nonlin", type=str, default='gelu',
                        help="non-linearity function to use [relu, elu, mish, softplus, tanh, gelu]")
    
    parser.add_argument("--optim", type=str, default='adam',
                        help="optimization algorithm to use [adam, sgd, rmsprop]")
    
    parser.add_argument("--save_every", type=int, default=25,
                        help="saves model results and weights every X epochs")
    
    parser.add_argument("--batch_size", type=int, default=32,
                        help="training batch size")
    
    parser.add_argument("--nsamples", type=int, default=4,
                        help="number of samples generated from hypernet")
    
    parser.add_argument("--norm", type=str, default='layer',
                        help="normalization method to use [layer, none]")
    
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate for T")
    
    parser.add_argument("--hnet_channels", type=int, default=3, 
                        help="number of channels for hypernet")
    
    parser.add_argument("--hnet_width", type=int, default=5,   
                        help="width of hypernet")
    
    parser.add_argument("--drugs", nargs='+', default=None, 
                        help="drug filters") 
    
    parser.add_argument("--cell_lines", nargs='+', default=None,
                        help="cell line filters")
    
    parser.add_argument("--test_prop", type=float, default=0.15,
                        help="proportion of data to assign to test set")
    
    parser.add_argument("--val_prop", type=float, default=0.05,
                        help="proportion of data to assign to test set")
    
    parser.add_argument("--checkpoint", action='store_true', default=False,
                        help="whether to checkpoint gradient (to reduce mem use)")
    
    parser.add_argument("--blur", type=float, default=0.05,
                        help="the sinkhorn blur parameter")
    
    parser.add_argument("--scaling", type=float, default=0.9,
                        help="the sinkhorn scaling parameter [0,1]; higher is more accurate but slower")
    
    
    args = parser.parse_args()

    return args

def eval(T, args, sampler, crit, partition='test'):

    T.eval()
    losses = []
    for cond_idx in range(len(sampler)): 
        print(f'evaluating condition {cond_idx}/{len(sampler)}', end='\r')

        with torch.no_grad(): 
            X,y = sampler.sample(cond_idx, batch_size=None, partition=partition)
            yhat = []
            for idx in torch.split(torch.arange(X.size(0)), args.batch_size): 
                XX = X[idx].to(device)
                yhat.append( T(XX) + XX[:, data.X2Y0_idxs] )
            yhat = torch.cat(yhat, dim=0)
            loss = crit(yhat, y.to(device))
            losses.append(loss.item())

    return np.mean(losses)



if __name__ == '__main__': 

    time0 = time.time() 

    # get args 
    args = get_args()

    os.makedirs(args.out, exist_ok=True)
    
    if torch.cuda.is_available():
        device = 'cuda'
        for i in range(torch.cuda.device_count()): print(f'cuda device {i}: {torch.cuda.get_device_properties(i).name}')
    else: 
        device = 'cpu'
    args.device = device
    print('using device:', device)
    
    data = torch.load(f'{args.data}/data.pt')

    sampler = scSampler(f'{args.data}/', 
                        drug_filter=args.drugs, 
                        cell_filter=args.cell_lines, 
                        test_prop=args.test_prop, 
                        val_prop=args.val_prop)

    T = GSNN(edge_index_dict                    = data.edge_index_dict, 
                node_names_dict                 = data.node_names_dict,
                channels                        = args.channels, 
                layers                          = args.layers, 
                dropout                         = args.dropout,
                residual                        = True,
                nonlin                          = torch.nn.GELU,
                bias                            = True,
                share_layers                    = True,
                add_function_self_edges         = True,
                checkpoint                      = args.checkpoint,      
                norm                            = 'layer').to(device)
    
    #T = HyperNet(gsnn, stochastic_channels=args.hnet_channels, width=args.hnet_width).to(device)

    crit = SamplesLoss(loss='sinkhorn', p=2, blur=args.blur, scaling=args.scaling, debias=True, reach=None, backend='online')
    eMMD = SamplesLoss(loss='energy')
    gMMD = SamplesLoss(loss='gaussian', blur=0.05)

    optim = torch.optim.Adam(T.parameters(), lr=args.lr)

    val_losses = []
    train_losses = []
    for i in range(args.epochs): 
        
        losses = [] 
        emmds = []
        gmmds = []
        for cond_idx in range(len(sampler)): 
            print(f'[batch progress: {cond_idx}/{len(sampler)}]', end='\r')
            T.train()

            X,y = sampler.sample(cond_idx, batch_size=args.batch_size, partition='train')

            X = X.to(device); y=y.to(device)
            y0 = X[:, data.X2Y0_idxs]

            optim.zero_grad()
            yhat = T(X) + y0 
            loss = crit(yhat, y)
            loss.backward()
            optim.step()
            losses.append(loss.item())

            with torch.no_grad():
                emmds.append(eMMD(yhat, y).item())
                gmmds.append(gMMD(yhat, y).item())

        val_losses.append( eval(T, args, sampler, crit, partition='val') ) 
        train_losses.append(np.mean(losses))
            
        torch.save({'sampler':sampler, 
                    'T':T, 
                    'args':args,
                    'train_losses':train_losses,
                    'val_losses':val_losses}, args.out + '/res_dict.pt')
        
        print(f'epoch: {i} ------> loss: {train_losses[-1]:.3f}, eMMD: {np.mean(emmds):.4f}, gMMD: {np.mean(gmmds):.5f} | val loss: {val_losses[-1]:.4f}')

    print()