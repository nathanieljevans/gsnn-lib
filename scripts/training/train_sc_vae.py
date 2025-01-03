import torch 
import argparse 
import os 
import numpy as np 
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, roc_auc_score
from sklearn import metrics
import pandas as pd
import sys 
from gsnn.models.VAE import VAE


import warnings
warnings.filterwarnings("ignore")

def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../../proc/sciplex3/',
                        help="path to data directory")

    parser.add_argument("--out", type=str, default='../../output/sciplex3/vae/',
                        help="path to output directory")
    
    parser.add_argument("--epochs", type=int, default=1000,
                        help="number of training iterations")
    
    parser.add_argument("--channels", type=int, default=512,
                        help="number of channels for each function node")
    
    parser.add_argument("--layers", type=int, default=1,
                        help="number of layers of message passing")
    
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate for the transport optimization")
        
    parser.add_argument("--batch_size", type=int, default=5000,
                        help="training batch size")
    
    parser.add_argument("--ignore_cuda", action='store_true',
                        help="whether to ignore available cuda GPU")
    
    parser.add_argument("--patience", type=int, default=100,
                        help="epoch patience for early stopping")
    
    parser.add_argument("--min_delta", type=float, default=0.0001,
                        help="minimum improvement for early stopping")
    
    parser.add_argument("--latent_dim", type=int, default=124,
                        help="latent dimension for autoencoder (not relevant to NN)")
    
    parser.add_argument("--beta", type=float, default=1e-4,
                        help="beta parameter for VAE loss")
    
    parser.add_argument("--test_prop", type=float, default=0.1,
                        help="proportion of data to use for testing")
    
    
    args = parser.parse_args()

    return args

def load_data(root): 

    fnames = [root + '/PROC/' + x for x in os.listdir(root + '/PROC/') if x.startswith('pert')]

    y = [] 
    for i, fname in enumerate(fnames): 
        print(f'loading data... {i+1}/{len(fnames)}', end='\r')
        y.append(torch.load(fname, weights_only=True))
    y = torch.stack(y, dim=0)

    return y 


if __name__ == '__main__': 
    args = get_args()
    print(args, "\n")

    y = load_data(args.data)
    print('y shape:', y.shape)

    device = 'cuda' if torch.cuda.is_available() and not args.ignore_cuda else 'cpu'

    model = VAE(input_dim=y.shape[1], latent_dim=args.latent_dim, num_layers=args.layers, hidden_channels=args.channels, dropout=0.)

    model.optimize(y, 
                   device=device, 
                   lr=args.lr, 
                   epochs=args.epochs, 
                   batch_size=args.batch_size, 
                   verbose=True, 
                   beta=args.beta, 
                   patience=args.patience, 
                   train_p=1-args.test_prop)
    
    os.makedirs(args.out, exist_ok=True)
    torch.save(model, os.path.join(args.out, 'scvae.pt'))
    