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
from gsnn.ot.utils import eval

import warnings
warnings.filterwarnings("ignore")

import gc

def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../sc_data/',
                        help="path to data directory")

    parser.add_argument("--out", type=str, default='../sc_output_not/',
                        help="path to output directory")
    
    parser.add_argument("--iters", type=int, default=1000,
                        help="number of training iterations")
    
    parser.add_argument("--channels", type=int, default=8,
                        help="number of channels for each function node")
    
    parser.add_argument("--layers", type=int, default=10,
                        help="number of layers of message passing")
    
    parser.add_argument("--dropout", type=float, default=0.,
                        help="")
        
    parser.add_argument("--save_every", type=int, default=10,
                        help="saves model results and weights every X epochs")
    
    parser.add_argument("--batch_size", type=int, default=64,
                        help="training batch size")
    
    parser.add_argument("--norm", type=str, default='layer',
                        help="GSNN normalization method to use [layer, none]")
    
    parser.add_argument("--wd", type=float, default=0.,
                        help="weight decay")
    
    parser.add_argument("--val_prop", type=float, default=0.1,
                        help="proportion of validation data")
    
    parser.add_argument("--test_prop", type=float, default=0.1,
                        help="proportion of test data")
    
    parser.add_argument("--method", type=str, default='shd',
                        help="transport optimization strategy to use [not, shd]")
    
    parser.add_argument("--T_lr", type=float, default=1e-3,
                        help="learning rate for T")
    
    parser.add_argument("--drugs", nargs='+', default=None, 
                        help="drug filters") 
    
    parser.add_argument("--cell_lines", nargs='+', default=None,
                        help="cell line filters")
    
    parser.add_argument("--f_lr", type=float, default=1e-4,
                        help="learning rate for f [relevant only if `method='not'`]")
    
    parser.add_argument("--f_channels", type=int, default=124,  
                         help="number of hidden channels for f [relevant only if `method='not'`]")
    
    parser.add_argument("--f_layers", type=int, default=2,  
                         help="number of hidden layers for f [relevant only if `method='not'`]")
    
    parser.add_argument("--f_iters", type=int, default=1,
                        help="number of iterations for f [relevant only if `method='not'`]")
    
    parser.add_argument("--T_iters", type=int, default=4,
                        help="number of iterations for T [relevant only if `method='not'`]")
    
    parser.add_argument("--blur", type=float, default=0.05,
                        help="the sinkhorn blur parameter [relevant only if `method='shd'`]")
    
    parser.add_argument("--scaling", type=float, default=0.9,
                        help="the sinkhorn scaling parameter [0,1]; higher is more accurate but slower [relevant only if `method='shd'`]")
    
    parser.add_argument("--reach", type=float, default=None,
                        help="the reach parameter for the sinkhorn distance; beneficial in noisy settings with outliers [relevant only if `method='shd'`]")

    parser.add_argument("--T_arch", type=str, default='gsnn',
                        help="architecture for T [gsnn, nn]")
    
    parser.add_argument("--checkpoint", action='store_true',
                        help="checkpoint GSNN layer gradients; will reduce memory but increase computation time")
    
    args = parser.parse_args()

    return args


if __name__ == '__main__': 

    time0 = time.time() 

    # get args 
    args = get_args()
    print(args); print()

    os.makedirs(args.out, exist_ok=True)
    
    data = torch.load(f'{args.data}/data.pt')

    sampler = scSampler(f'{args.data}/',
                        drug_filter=args.drugs,
                        cell_filter=args.cell_lines,
                        test_prop=args.test_prop,
                        val_prop=args.val_prop)
    
    if args.method == 'not': 
        trainer = NOT(args, data, sampler)
    elif args.method == 'shd': 
        trainer = SHD(args, data, sampler)
    else:
        raise ValueError('method not recognized [shd, not]')

    mmds = []; shds = []; wasss = []; mu_r2 = []
    for i in range(args.iters): 

        loss = trainer.step(sampler)
        mmd_, shd_, wass_, mu_r2_ = eval(trainer.get_T(), sampler, batch_size=args.batch_size, partition='val', max_n=args.batch_size)
        mmds.append(mmd_); shds.append(shd_); wasss.append(wass_); mu_r2.append(mu_r2_)

        if (i % args.save_every) == 0:
            save_dict = {**trainer.state_dict(), **{'sampler':sampler, 
                                                    'args':args,
                                                    'mmds':mmds,
                                                    'shds':shds,
                                                    'wass':wasss,
                                                    'mu_r2':mu_r2}}
            torch.save(save_dict, f'{args.out}/res_dict.pt')
        
        print(f'iters: {i} ---> train loss: {loss:.3f} | val mmd: {mmds[-1]:.3f} | val shd: {shds[-1]:.3f} | val wass:{wasss[-1]:.3f} | val mu r2: {mu_r2[-1]:.3f}')