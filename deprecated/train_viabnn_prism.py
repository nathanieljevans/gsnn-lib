

import torch 
import argparse
import numpy as np 
import pickle as pkl 
from torch.utils.data import DataLoader
import pandas as pd 
from sklearn.metrics import r2_score 
from matplotlib import pyplot as plt 
import os 

import sys 
sys.path.append('../.')
from gsnn_lib.data.PrismDataset import PrismDataset
from gsnn.models.utils import predict_gsnn
from gsnn.models.GSNN import GSNN 
from gsnn.models.NN import NN 
from gsnn_lib.proc.prism.utils import load_prism
from gsnn.optim.EarlyStopper import EarlyStopper


from hnet.models import HyperNet
from hnet.train.hnet import train_hnet
from hnet.models.MLP import MLP 
from hnet.models.HyperNet import HyperNet
from hnet.train.hnet import EnergyDistanceLoss
 
torch.multiprocessing.set_sharing_strategy('file_system')

def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default='../../../data/',
                        help="path to data directory")
    
    parser.add_argument("--proc", type=str, default='../../proc/lincs/',
                        help="path to processed data dir")
    
    parser.add_argument("--partition_dir", type=str, default='/partitions/',
                        help="path to partition directory")
    
    parser.add_argument("--uid_dir", type=str, default=None,
                        help="directory containing the trained models (one for each fold)")
    
    parser.add_argument("--expr_batch", type=int, default=256,
                        help="GSNN batch size to use")
    
    parser.add_argument("--batch", type=int, default=4096,
                        help="batch size to use while training cell viab predictor")
    
    parser.add_argument("--epochs", type=int, default=250,
                        help="number of epochs to train")
    
    parser.add_argument("--channels", type=int, default=256,
                        help="cell viab predictor nn hidden channels")
    
    parser.add_argument("--layers", type=int, default=3,
                        help="cell viab predictor nn layers")
    
    parser.add_argument("--workers", type=int, default=12,
                        help="number of workers to use for the data loader when predicting transcriptional activations")
    
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout rate to use while training cell viab predictor")
    
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate to use for cell viab predictor")
    
    parser.add_argument("--patience", type=int, default=50,
                        help="early stopping patience")
    
    parser.add_argument("--min_delta", type=float, default=1e-3,
                        help="early stopping min delta")
    
    return parser.parse_args()

def train_hypernet(hnet, x_train, y_train, x_val, y_val, lr=1e-3, epochs=100, batch_size=512, 
                   patience=10, min_delta=1e-3): 

    # train hypernet 
    optim = torch.optim.Adam(hnet.parameters(), lr=lr)
    crit = torch.nn.MSELoss()
    
    stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    best_r2 = -np.inf
    best_state_dict = None 
    for epoch in range(epochs): 

        losses = [] 
        splits = torch.split(torch.randperm(x_train.size(0)), batch_size)
        hnet.train()
        for ii, ixs in enumerate(splits): 
            optim.zero_grad()
            yhat = hnet(x_train[ixs].to(device)).sigmoid()
            
            loss = crit(yhat, y_train[ixs].to(device))
            
            loss.backward()
            optim.step()
            losses.append(loss.item())
            print(f'[{ii}/{len(splits)}]->{loss.item():.2f}', end='\r')

        train_loss = np.mean(losses)

        hnet.eval()
        val_yhat = [] 
        for ixs in torch.split(torch.arange(x_val.size(0)), batch_size):
            with torch.no_grad(): 
                val_yhat.append( hnet(x_val[ixs].to(device)).sigmoid().detach().cpu() )
        val_yhat = torch.cat(val_yhat, 0)

        val_r2 = r2_score(y_val.detach().cpu().numpy().ravel(), val_yhat.detach().cpu().numpy().ravel())

        if val_r2 >= best_r2: 
            best_r2 = val_r2
            best_state_dict = hnet.state_dict()

        print(f'epoch: {epoch}... train loss: {train_loss:.3f}... val r2: {val_r2:.3f}')

        if stopper.early_stop(-val_r2): 
            print('########################################')
            print(f'early stopping @ epoch: {epoch}... best val r2: {best_r2:.2f}')
            print('########################################')
            break
    
    hnet.load_state_dict(best_state_dict)
    hnet.eval()

    return hnet


if __name__ == '__main__': 

    # get args 
    args = get_args()

    print()
    print(args)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    splits = np.sort([x for x in os.listdir(f'{args.proc}/{args.partition_dir}') if 'prism_splits' in x])  # where the prism ids partition splits are saved 
    fold_dirs = np.sort([x.split('.')[0] for x in [y for y in os.listdir(args.uid_dir) if 'fold' in y]]) # where the expr models are saved 

    metric_dicts = [] 
    for split,fold_dir in zip(splits, fold_dirs):

        print() 
        print('#'*50)

        # out dir should be the same location as the expr model 
        out_dir = f'{args.uid_dir}/{fold_dir}/'
        os.makedirs(out_dir, exist_ok=True)
        print('prism split dir:', split)
        print('output directory:', out_dir)

        with open(f'{out_dir}/cellviab_args.log', 'w') as f: 
            f.write(str(args))

        data = torch.load(f'{out_dir}/Data.pt', weights_only=False)
        model = torch.load(f'{out_dir}/best_model.pt', weights_only=False).eval().to(device)

        partition_dict = torch.load(f'{args.proc}/{args.partition_dir}/fold_0_prism_splits.pt', weights_only=False) # TODO loop this ... 
        prism_train_ids = partition_dict['train_obs']
        prism_test_ids = partition_dict['test_obs']
        prism_val_ids = partition_dict['val_obs']

        all_obs = prism_train_ids.tolist() + prism_test_ids.tolist() + prism_val_ids.tolist()

        train_ixs = torch.arange(len(prism_train_ids))
        test_ixs = torch.arange(len(prism_train_ids), len(prism_train_ids) + len(prism_test_ids))
        val_ixs = torch.arange(len(prism_train_ids) + len(prism_test_ids), len(prism_train_ids) + len(prism_test_ids) + len(prism_val_ids))

        # prism datasets 
        prism       = load_prism(args.data, cellspace=data.cellspace, drugspace=data.drugspace)
        dataset     = PrismDataset(prism, prism_ids=all_obs, data=data, clamp=True)
        loader      = DataLoader(dataset, batch_size=args.expr_batch, num_workers=args.workers)

        # predict on all data
        x_expr = [] 
        y_viab = [] 
        
        with torch.no_grad(): 
            for i,(x, y, *sig_id) in enumerate(loader): 
                print(f'progress: {i}/{len(loader)}', end='\r')
                x_expr.append( model(x.to(device)).detach().cpu() )
                y_viab.append( y.detach().cpu() )

        x = torch.cat(x_expr, 0)
        y = torch.cat(y_viab, 0)
        
        x_train = x[train_ixs].squeeze(-1)
        x_val = x[val_ixs].squeeze(-1)
        x_test = x[test_ixs].squeeze(-1)

        y_train = y[train_ixs].view(-1,1)
        y_val = y[val_ixs].view(-1,1)
        y_test = y[test_ixs].view(-1,1)


        print('train set:', x_train.size())
        print('val set:', x_val.size())
        print('test set:', x_test.size())

        ################# train hnet ####################
        mlp_kwargs = {'hidden_channels':args.channels, 
                            'layers':args.layers, 
                            'nonlin':'elu', 
                            }

        mlp = MLP(in_channels=x.size(1), out_channels=1, **mlp_kwargs).to(device)
        #mlp = MLP(in_channels=x.size(1), out_channels=1, **mlp_kwargs)
        #hnet = HyperNet(mlp, **hnet_kwargs).to(device).to(device)

        n_params = sum([p.numel() for p in mlp.parameters()])
        print('# params:', n_params)

        print('training cell viab predictor...')
        torch.cuda.empty_cache()
        mlp = train_hypernet(mlp, x_train, y_train, x_val, y_val, lr=args.lr, epochs=args.epochs, 
                            batch_size=args.batch, patience=args.patience, min_delta=args.min_delta)

        print() 

        # predict on validation set
        mlp.eval()

        yhat = [] 
        for ixs in torch.split(torch.arange(x_val.size(0)), args.batch): 
            with torch.no_grad(): 
                yhat.append( mlp(x_val[ixs].to(device)).sigmoid().detach().cpu() )  # samples, batch, 1 
        yhat = torch.cat(yhat, 0) # avg over samples

        r2_val = r2_score(y_val.detach().cpu().numpy().ravel(), yhat.detach().cpu().numpy().ravel()) 
        r_val = np.corrcoef(y_val.detach().cpu().numpy().ravel(), yhat.detach().cpu().numpy().ravel())[0,1]
        mse_val = ((y_val.ravel() - yhat.ravel())**2).mean().detach().cpu().item()

        print()
        print('validation set performance: ')
        print('\tR2: ', r2_val)
        print('\tR: ', r_val)
        print('\tMSE: ', mse_val)
        print() 

        # predict on test set
        yhat = []
        for ixs in torch.split(torch.arange(x_test.size(0)), args.batch): 
            with torch.no_grad(): 
                yhat.append( mlp(x_test[ixs].to(device)).sigmoid().detach().cpu() )
        yhat = torch.cat(yhat, 0)

        r2_test = r2_score(y_test.detach().cpu().numpy(), yhat.detach().cpu().numpy())
        r_test = np.corrcoef(y_test.detach().cpu().numpy().ravel(), yhat.detach().cpu().numpy().ravel())[0,1]
        mse_test = ((y_test - yhat)**2).mean().detach().cpu().item()

        print()
        print('test set performance: ')
        print('\tR2: ', r2_test)
        print('\tR: ', r_test)
        print('\tMSE: ', mse_test)
        print()

        metric_dict = {'r2_val': r2_val, 'r_val': r_val, 'mse_val': mse_val, 'r2_test': r2_test, 'r_test': r_test, 'mse_test': mse_test}
        metric_dicts.append(metric_dict)

        torch.save(mlp, f'{out_dir}/ViabPredictor.pt')
        torch.save(metric_dict, f'{out_dir}/ViabPredictorMetrics.pt')
        torch.save(yhat, f'{out_dir}/ViabTestPredictions.pt')

    print('#'*50)
    print('#'*50)
    avg_metric = {k:np.mean([x[k] for x in metric_dicts]) for k in metric_dicts[0].keys()}
    print('average performance across folds: ')
    print(avg_metric)
    print()

    torch.save(avg_metric, f'{args.uid_dir}/avg_viab_metric_dict.pt')


'''

import torch 
import argparse
import numpy as np 
import pickle as pkl 
from torch.utils.data import DataLoader
import pandas as pd 
from sklearn.metrics import r2_score 
from matplotlib import pyplot as plt 
import os 

import sys 
sys.path.append('../.')
from gsnn_lib.data.PrismDataset import PrismDataset
from gsnn.models.utils import predict_gsnn
from gsnn.models.GSNN import GSNN 
from gsnn.models.NN import NN 
from gsnn_lib.proc.prism.utils import load_prism
from gsnn.optim.EarlyStopper import EarlyStopper


from hnet.models import HyperNet
from hnet.train.hnet import train_hnet
from hnet.models.MLP import MLP 
from hnet.models.HyperNet import HyperNet
from hnet.train.hnet import EnergyDistanceLoss
 
torch.multiprocessing.set_sharing_strategy('file_system')

def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default='../../../data/',
                        help="path to data directory")
    
    parser.add_argument("--proc", type=str, default='../../proc/lincs/',
                        help="path to processed data dir")
    
    parser.add_argument("--partition_dir", type=str, default='/partitions/',
                        help="path to partition directory")
    
    parser.add_argument("--uid_dir", type=str, default=None,
                        help="directory containing the trained models (one for each fold)")
    
    parser.add_argument("--expr_batch", type=int, default=200,
                        help="GSNN batch size to use")
    
    parser.add_argument("--batch", type=int, default=1000,
                        help="batch size to use while training cell viab predictor")
    
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs to train")
    
    parser.add_argument("--channels", type=int, default=64,
                        help="cell viab predictor nn hidden channels")
    
    parser.add_argument("--layers", type=int, default=1,
                        help="cell viab predictor nn layers")
    
    parser.add_argument("--workers", type=int, default=12,
                        help="number of workers to use for the data loader when predicting transcriptional activations")
    
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout rate to use while training cell viab predictor")
    
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate to use for cell viab predictor")
    
    parser.add_argument("--loss_fn", type=str, default='edl',
                        help="loss function to use for training cell viab predictor")
    
    parser.add_argument("--nsamples", type=int, default=250,
                        help="number of samples to use for training hnet")
    
    parser.add_argument("--stochastic_channels", type=int, default=4,
                        help="number of stochastic channels to use for hnet")
    
    parser.add_argument("--hnet_width", type=int, default=100,
                        help="width of the hnet")
    
    parser.add_argument("--patience", type=int, default=10,
                        help="early stopping patience")
    
    parser.add_argument("--min_delta", type=float, default=1e-3,
                        help="early stopping min delta")
    
    return parser.parse_args()

def train_hypernet(hnet, x_train, y_train, x_val, y_val, nsamples=100, lr=1e-3, epochs=100, batch_size=512, 
                   patience=10, min_delta=1e-3, loss_fn='nll'): 

    # train hypernet 
    hnet.train()
    optim = torch.optim.Adam(hnet.parameters(), lr=lr)
    
    if loss_fn == 'mse': 
        crit = torch.nn.MSELoss()
    elif loss_fn == 'l1': 
        crit = torch.nn.SmoothL1Loss()
    elif loss_fn == 'nll': 
        crit = torch.nn.GaussianNLLLoss()
    elif loss_fn in ['edl']: 
        crit = EnergyDistanceLoss()
    else:
        raise NotImplementedError('unrecognized loss string, options: [mse, l1, nll, edl, ce]')
    
    stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    best_r2 = -np.inf
    best_state_dict = None 
    for epoch in range(epochs): 

        losses = [] 
        splits = torch.split(torch.randperm(x_train.size(0)), batch_size)
        for ii, ixs in enumerate(splits): 
            optim.zero_grad()
            yhat = hnet(x_train[ixs].to(device), samples=nsamples).sigmoid()
            
            if loss_fn in ['mse', 'l1']: 
                loss = crit(yhat, y_train[ixs].to(device).unsqueeze(0).expand(nsamples,-1,-1))
            elif loss_fn in ['edl']: 
                loss = crit(yhat, y_train[ixs].to(device))
            elif loss_fn == 'nll': 
                loss = crit(yhat.mean(0), y_train[ixs].to(device), yhat.var(0) )
            else: 
                raise Exception('no loss objective defined')

            loss.backward()
            optim.step()
            losses.append(loss.item())
            print(f'[{ii}/{len(splits)}]->{loss.item():.2f}', end='\r')

        train_loss = np.mean(losses)

        val_yhat = [] 
        for ixs in torch.split(torch.arange(x_val.size(0)), batch_size):
            with torch.no_grad(): 
                val_yhat.append( hnet(x_val[ixs].to(device), samples=nsamples).detach().cpu() )
        val_yhat = torch.cat(val_yhat, 1).mean(0)

        val_r2 = r2_score(y_val.detach().cpu().numpy().ravel(), val_yhat.detach().cpu().numpy().ravel())

        if val_r2 >= best_r2: 
            best_r2 = val_r2
            best_state_dict = hnet.state_dict()

        print(f'epoch: {epoch}... train loss: {train_loss:.3f}... val r2: {val_r2:.3f}')

        if stopper.early_stop(-val_r2): 
            print('########################################')
            print(f'early stopping @ epoch: {epoch}... best val r2: {best_r2:.2f}')
            print('########################################')
            break
    
    hnet.load_state_dict(best_state_dict)

    return hnet


if __name__ == '__main__': 

    # get args 
    args = get_args()

    print()
    print(args)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    splits = [x for x in os.listdir(f'{args.proc}/{args.partition_dir}') if 'prism_splits' in x]  # where the prism ids partition splits are saved 
    fold_dirs = [x.split('.')[0] for x in [y for y in os.listdir(args.uid_dir) if 'fold' in y]] # where the expr models are saved 

    metric_dicts = [] 
    for split,fold_dir in zip(splits, fold_dirs):

        print() 
        print('#'*50)

        # out dir should be the same location as the expr model 
        out_dir = f'{args.uid_dir}/{fold_dir}/'
        os.makedirs(out_dir, exist_ok=True)
        print('prism split dir:', split)
        print('output directory:', out_dir)

        with open(f'{out_dir}/cellviab_args.log', 'w') as f: 
            f.write(str(args))

        data = torch.load(f'{out_dir}/Data.pt', weights_only=False)
        model = torch.load(f'{out_dir}/best_model.pt', weights_only=False).eval().to(device)

        partition_dict = torch.load(f'{args.proc}/{args.partition_dir}/fold_0_prism_splits.pt', weights_only=False) # TODO loop this ... 
        prism_train_ids = partition_dict['train_obs']
        prism_test_ids = partition_dict['test_obs']
        prism_val_ids = partition_dict['val_obs']

        all_obs = prism_train_ids.tolist() + prism_test_ids.tolist() + prism_val_ids.tolist()

        train_ixs = torch.arange(len(prism_train_ids))
        test_ixs = torch.arange(len(prism_train_ids), len(prism_train_ids) + len(prism_test_ids))
        val_ixs = torch.arange(len(prism_train_ids) + len(prism_test_ids), len(prism_train_ids) + len(prism_test_ids) + len(prism_val_ids))

        # prism datasets 
        prism       = load_prism(args.data, cellspace=data.cellspace, drugspace=data.drugspace)
        dataset     = PrismDataset(prism, prism_ids=all_obs, data=data, clamp=True)
        loader      = DataLoader(dataset, batch_size=args.expr_batch, num_workers=args.workers)

        # predict on all data
        x_expr = [] 
        y_viab = [] 
        
        with torch.no_grad(): 
            for i,(x, y, *sig_id) in enumerate(loader): 
                print(f'progress: {i}/{len(loader)}', end='\r')
                x_expr.append( model(x.to(device)).detach().cpu() )
                y_viab.append( y.detach().cpu() )

        x = torch.cat(x_expr, 0)
        y = torch.cat(y_viab, 0)
        
        x_train = x[train_ixs].squeeze(-1)
        x_val = x[val_ixs].squeeze(-1)
        x_test = x[test_ixs].squeeze(-1)

        y_train = y[train_ixs].view(-1,1)
        y_val = y[val_ixs].view(-1,1)
        y_test = y[test_ixs].view(-1,1)


        print('train set:', x_train.size())
        print('val set:', x_val.size())
        print('test set:', x_test.size())

        ################# train hnet ####################
        mlp_kwargs = {'hidden_channels':args.channels, 
                            'layers':args.layers, 
                            'nonlin':'elu', 
                            }

        hnet_kwargs = {'stochastic_channels':args.stochastic_channels, 
                        'width':args.hnet_width}
        
        model = MLP(in_channels=x.size(1), out_channels=1, **mlp_kwargs)
        #mlp = MLP(in_channels=x.size(1), out_channels=1, **mlp_kwargs)
        #hnet = HyperNet(mlp, **hnet_kwargs).to(device).to(device)

        n_params = sum([p.numel() for p in hnet.parameters()])
        print('# params:', n_params)

        print('training cell viab predictor...')
        torch.cuda.empty_cache()
        hnet = train_hypernet(hnet, x_train, y_train, x_val, y_val, lr=args.lr, epochs=args.epochs, 
                            batch_size=args.batch, patience=args.patience, min_delta=args.min_delta,
                            nsamples=args.nsamples)

        print() 

        # predict on validation set
        hnet.eval()

        yhat = [] 
        for ixs in torch.split(torch.arange(x_val.size(0)), args.batch): 
            with torch.no_grad(): 
                yhat.append( hnet(x_val[ixs].to(device), samples=args.nsamples).detach().cpu() )  # samples, batch, 1 
        yhat = torch.cat(yhat, 1).mean(0) # avg over samples

        r2_val = r2_score(y_val.detach().cpu().numpy().ravel(), yhat.detach().cpu().numpy().ravel()) 
        r_val = np.corrcoef(y_val.detach().cpu().numpy().ravel(), yhat.detach().cpu().numpy().ravel())[0,1]
        mse_val = ((y_val.ravel() - yhat.ravel())**2).mean().detach().cpu().item()

        print()
        print('validation set performance: ')
        print('\tR2: ', r2_val)
        print('\tR: ', r_val)
        print('\tMSE: ', mse_val)
        print() 

        # predict on test set
        yhat = []
        for ixs in torch.split(torch.arange(x_test.size(0)), args.batch): 
            with torch.no_grad(): 
                yhat.append( hnet(x_test[ixs].to(device), samples=args.nsamples).detach().cpu() )
        yhat = torch.cat(yhat, 1).mean(0) # avg over samples

        r2_test = r2_score(y_test.detach().cpu().numpy(), yhat.detach().cpu().numpy())
        r_test = np.corrcoef(y_test.detach().cpu().numpy().ravel(), yhat.detach().cpu().numpy().ravel())[0,1]
        mse_test = ((y_test - yhat)**2).mean().detach().cpu().item()

        print()
        print('test set performance: ')
        print('\tR2: ', r2_test)
        print('\tR: ', r_test)
        print('\tMSE: ', mse_test)
        print()

        metric_dict = {'r2_val': r2_val, 'r_val': r_val, 'mse_val': mse_val, 'r2_test': r2_test, 'r_test': r_test, 'mse_test': mse_test}
        metric_dicts.append(metric_dict)

        torch.save(hnet, f'{out_dir}/ViabPredictor.pt')
        torch.save(metric_dict, f'{out_dir}/ViabPredictorMetrics.pt')
        torch.save(yhat, f'{out_dir}/ViabTestPredictions.pt')

    print('#'*50)
    print('#'*50)
    avg_metric = {k:np.mean([x[k] for x in metric_dicts]) for k in metric_dicts[0].keys()}
    print('average performance across folds: ')
    print(avg_metric)
    print()

    torch.save(avg_metric, f'{args.uid_dir}/avg_viab_metric_dict.pt')


    

'''