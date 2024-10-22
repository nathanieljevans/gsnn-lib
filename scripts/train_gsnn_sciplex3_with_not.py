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

    parser.add_argument("--out", type=str, default='../sc_output_not/',
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
    
    parser.add_argument("--batch_size", type=int, default=100,
                        help="training batch size")
    
    parser.add_argument("--T_iters", type=int, default=4,
                        help="number of transport iterations per batch")
    
    parser.add_argument("--f_iters", type=int, default=1,
                        help="number of transport iterations per batch")
    
    parser.add_argument("--norm", type=str, default='layer',
                        help="normalization method to use [layer, none]")
    
    parser.add_argument("--f_channels", type=int, default=100,  
                         help="number of hidden channels for f")
    
    parser.add_argument("--f_layers", type=int, default=2,  
                         help="number of hidden layers for f")
    
    parser.add_argument("--T_lr", type=float, default=1e-4,
                        help="learning rate for T")
    
    parser.add_argument("--f_lr", type=float, default=1e-4,
                        help="learning rate for f")
    
    parser.add_argument("--drugs", nargs='+', default=None, 
                        help="drug filters") 
    
    parser.add_argument("--cell_lines", nargs='+', default=None,
                        help="cell line filters")
    
    args = parser.parse_args()

    return args

def freeze_(model):
    for param in model.parameters():
        param.requires_grad = False
    model = model.eval()

def unfreeze_(model):
    for param in model.parameters():
        param.requires_grad = True
    model = model.train()

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

    sampler = scSampler(f'{args.data}/', drug_filter=args.drugs, cell_filter=args.cell_lines)

    T = GSNN(edge_index_dict                 = data.edge_index_dict, 
                 node_names_dict                = data.node_names_dict,
                channels                        = args.channels, 
                layers                          = args.layers, 
                dropout                         = args.dropout,
                residual                        = True,
                nonlin                          = torch.nn.GELU,
                bias                            = True,
                share_layers                    = False,
                add_function_self_edges         = True,
                norm                            = 'layer').to(device)
    
    # T = HyperNet(gsnn, stochastic_channels=args.hnet_channels, width=args.hnet_width).to(device)

    # OT cost function
    #  TODO: need to one hot encode condition or use separate critic networks 
    '''f_dict = {i:NN(in_channels = len(data.node_names_dict['output']), 
                hidden_channels = args.f_channels, 
                out_channels = 1, 
                layers = args.f_layers, 
                norm = torch.nn.LayerNorm).to(device) for i in range(len(sampler))}'''
    
    f = NN(in_channels = len(data.node_names_dict['input']),
            hidden_channels = args.f_channels, 
            out_channels = 1, 
            layers = args.f_layers, 
            norm = torch.nn.LayerNorm).to(device)

    t_optim = torch.optim.Adam(T.parameters(), lr=args.T_lr, weight_decay=1e-10)
    f_optim = torch.optim.Adam(f.parameters(), lr=args.f_lr, weight_decay=1e-10)

    MSE = torch.nn.MSELoss()

    eMMD = SamplesLoss(loss='energy')

    unfreeze_(T) # should be unnecessary but jic 
    for i in range(args.epochs): 

        t_losses = [] 
        f_losses = [] 
        mmds = [] 
        for cond_idx in range(len(sampler)): 

            unfreeze_(T); freeze_(f)
            X,y = sampler.sample(cond_idx, batch_size=args.batch_size)
            X = X.to(device); y=y.to(device)
            y0 = X[:, data.X2Y0_idxs]
            
            # optimize T 
            for j in range(args.T_iters): 
                t_optim.zero_grad()
                yhat = T(X) + y0 
                yhatX = X.clone().detach()
                yhatX[:, data.X2Y0_idxs] = yhat

                cost_loss = MSE(yhat, y0) 
                critic_loss = f(yhatX).mean()
                t_loss = cost_loss - critic_loss

                t_loss.backward() 
                t_optim.step() 
                t_losses.append(t_loss.item())
                print(f'epoch: {i}, batch: {cond_idx} --> cost: {cost_loss.item():.3f}, critic: {critic_loss.item():.3f}', end='\r')


            #optimize F 
            freeze_(T); unfreeze_(f)
            for k in range(args.f_iters):
                X,y = sampler.sample(cond_idx, batch_size=args.batch_size)
                X = X.to(device); y=y.to(device)
                y0 = X[:, data.X2Y0_idxs]

                with torch.no_grad():
                    yhat = T(X) + y0
                    yhatX = X.clone().detach()
                    yhatX[:, data.X2Y0_idxs] = yhat
                    yX = X.clone().detach()
                    yX[:, data.X2Y0_idxs] = y

                f_optim.zero_grad() 
                yhat_score = f(yhatX).view(-1)
                y_score = f(yX).view(-1)
                f_loss = yhat_score.mean() - y_score.mean()
                f_loss.backward()
                f_optim.step()
                f_losses.append(f_loss.item())

            with torch.no_grad(): 
                mmd_ = eMMD(yhat.detach(), y.detach()).item()
                mmds.append(mmd_)
                label = torch.cat((torch.zeros_like(yhat_score), torch.ones_like(y_score)), dim=-1)
                score = torch.cat((yhat_score, y_score), dim=-1)
                auroc = roc_auc_score(label.detach().cpu().numpy(), score.detach().cpu().numpy())

                print(f'epoch: {i}, batch: {cond_idx} --> t_loss: {t_losses[-1]:.3f}, f_loss: {f_losses[-1]:.3f}, mmd: {mmd_:.4f}, auroc: {auroc:.3f}', end='\r')

        torch.save({'sampler':sampler, 
                    'T':T, 
                    'f':f,
                    'args':args,
                    'f_losses':f_losses,
                    't_losses':t_losses,
                    'mmds':mmds}, args.out + '/res_dict.pt')
        
        print(f'epoch: {i} ----------------------------------> T loss: {np.mean(t_losses):.3f} | f loss: {np.mean(f_losses):.3f}  | mmd: {np.mean(mmds):.3f}')



'''





def pretrain_f(args, T, data, f_dict, f_dict_optim, sampler, device):
    
    #freeze_(T) 
    # pretrain f
    for i in range(args.f_warmup): 
        aurocs = []
        for cond_idx in range(len(sampler)): 
            X,y = sampler.sample(cond_idx, batch_size=args.batch_size)
            X = X.to(device); y=y.to(device)
            y0 = X[:, data.X2Y0_idxs]
            
            #with torch.no_grad():
            #    yhat = T(X, samples=1).squeeze(0) + y0
           
            yhat = y + torch.randn_like(y)

            f_dict_optim[cond_idx].zero_grad() 
            yhat_score = f_dict[cond_idx](yhat).view(-1)
            y_score = f_dict[cond_idx](y).view(-1)
            y0_score = f_dict[cond_idx](y0).view(-1)

            reg_loss = (yhat_score**2).mean() + (y_score**2).mean() #+ (y0_score**2).mean()
            f_loss =  yhat_score.mean() - y_score.mean() + reg_loss # y0_score.mean() +
            f_loss.backward()
            f_dict_optim[cond_idx].step()
            with torch.no_grad(): 
                label = torch.cat((torch.zeros_like(yhat_score), torch.ones_like(y_score)), dim=-1)
                score = torch.cat((yhat_score, y_score), dim=-1)
                aurocs.append( roc_auc_score(label.detach().cpu().numpy(), score.detach().cpu().numpy()) )
        
        print(f'pretraining f... [{i}] --> auroc: {np.mean(aurocs):.3f}', end='\r')
    print()
    return f_dict




t_losses = [] 
        f_losses = [] 
        mmds = [] 
        for cond_idx in range(len(sampler)): 

            X,y = sampler.sample(cond_idx, batch_size=args.batch_size)
            X = X.to(device); y=y.to(device)
            y0 = X[:, data.X2Y0_idxs]
            yhat = T(X, samples=args.nsamples) + y0

            #optimize F 
            unfreeze_(f_dict[cond_idx])
            f_dict_optim[cond_idx].zero_grad() 
            yhat_score = f_dict[cond_idx](yhat.detach().view(-1, yhat.size(-1))).view(-1)
            y_score = f_dict[cond_idx](y).view(-1)
            f_loss = yhat_score.mean() - y_score.mean()
            f_loss.backward()
            f_dict_optim[cond_idx].step()
            f_losses.append(f_loss.item())

            freeze_(f_dict[cond_idx])
            t_optim.zero_grad()
            cost_loss = MSE(yhat, y0.unsqueeze(0).expand(args.nsamples, -1,-1)) 
            var_loss = args.gamma*yhat.var(dim=0).mean()
            critic_loss = f_dict[cond_idx](yhat.view(-1,yhat.size(-1))).mean()
            com_loss = torch.mean((yhat.mean(dim=1).mean() - y.mean(dim=0).mean().unsqueeze(0).expand(args.nsamples, -1))**2) # center of mass loss # this could be an energy distance as well
            t_loss = cost_loss - var_loss - critic_loss + com_loss 
            t_loss.backward() 
            t_optim.step() 
            t_losses.append(t_loss.item())

            with torch.no_grad(): 
                mmd_ = mmd(yhat.detach().cpu().numpy().mean(0), y.detach().cpu().numpy())
                mmds.append(mmd_)
                label = torch.cat((torch.zeros_like(yhat_score), torch.ones_like(y_score)), dim=-1)
                score = torch.cat((yhat_score, y_score), dim=-1)
                auroc = roc_auc_score(label.detach().cpu().numpy(), score.detach().cpu().numpy())

                print(f'epoch: {i}, batch: {cond_idx} --> t_loss: {t_losses[-1]:.3f}, f_loss: {f_losses[-1]:.3f}, mmd: {mmd_:.4f}, auroc: {auroc:.3f}', end='\r')

        torch.save({'sampler':sampler, 
                    'T':T, 
                    'f_dict':f_dict,
                    'args':args}, args.out + '/res_dict.pt')
        print(f'epoch: {i} ----------------------------------> T loss: {np.mean(t_losses):.3f} | f loss: {np.mean(f_losses):.3f}  | mmd: {np.mean(mmds):.3f}')


        
        
def initialize(T, eps=1e-1): 

    init_dict = {}
    for i,blk in enumerate(T.model.ResBlocks): 
        init_dict[f'ResBlocks.{i}.lin1.values'] = (torch.zeros_like(blk.lin1.init_var.cpu()), blk.lin1.init_var.cpu())
        init_dict[f'ResBlocks.{i}.lin3.values'] = (torch.zeros_like(blk.lin3.init_var.cpu()), blk.lin3.init_var.cpu())
        init_dict[f'ResBlocks.{i}.lin1.bias'] = (eps*torch.ones_like(blk.lin1.bias.cpu()), eps*torch.ones_like(blk.lin1.bias.cpu()))
        init_dict[f'ResBlocks.{i}.lin3.bias'] = (eps*torch.ones_like(blk.lin3.bias.cpu()), eps*torch.ones_like(blk.lin3.bias.cpu()))
        gamma = blk.norm1.gamma.cpu()
        init_dict[f'ResBlocks.{i}.norm1.gamma'] = (torch.ones_like(gamma), eps*torch.ones_like(gamma))
        beta  = blk.norm1.beta.cpu()
        init_dict[f'ResBlocks.{i}.norm1.beta'] = (eps*torch.ones_like(beta), eps*torch.ones_like(beta))
        
    T = init_hnet(T.cpu(), init_dict, samples=50, iters=50, lr=1e-1)

    return T
'''