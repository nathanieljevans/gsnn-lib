
import torch 
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data import DataLoader
import pandas as pd 
import argparse 
import numpy as np
import time 
import torch_geometric as pyg 
from sklearn.metrics import roc_auc_score

from gsnn.optim.Actor import GSNNActor
from gsnn.optim.SurrogateEnvironment import SurrogateEnvironment
from gsnn.optim.Environment import Environment
from gsnn.models.GSNN import GSNN
from gsnn_lib.data.LincsDataset import LincsDataset
from gsnn.models import utils 
from gsnn_lib.proc.lincs.utils import get_x_drug_conc           # required to unpickle data 
from torch_geometric.utils import to_undirected 
from gsnn.optim.BayesOpt import BayesOpt

import os


from gsnn_lib.utils.augment_edge_index import augment_edge_index


def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../processed_data/',
                        help="path to data directory")

    parser.add_argument("--fold", type=str, default='../processed_data/',
                        help="path to data fold directory; must contain data splits - see `create_data_splits.py`")
    
    parser.add_argument("--out", type=str, default='../output/',
                        help="output directory")
    
    parser.add_argument("--siginfo", type=str, default='../../data/',
                        help="path to siginfo directory")
    
    # GSNN parameters 
    parser.add_argument("--channels", type=int, default=5,
                        help="GSNN hidden channels")
    parser.add_argument("--layers", type=int, default=10,
                        help="GSNN number of layers")
    parser.add_argument("--dropout", type=float, default=0.,
                        help="GSNN dropout probability")
    parser.add_argument("--share_layers", action='store_true',
                        help="Whether GSNN should share parameters across layers")
    parser.add_argument("--add_function_self_edges", action='store_true',
                        help="Whether GSNN should add self loops to the graph")
    parser.add_argument("--norm", type=str, default='layer',
                        help="GSNN normalization method")
    parser.add_argument("--checkpoint", action='store_true',
                        help="whether to use gradient checkpointing for the GSNN model")
    
    # GSNN training parameters
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="GSNN learning rate")
    parser.add_argument("--max_epochs", type=int, default=25,
                        help="GSNN number of layers")
    parser.add_argument("--patience", type=int, default=5,
                        help="early stopping patience")
    parser.add_argument("--min_delta", type=float, default=1e-2,
                        help="early stopping minimum improvement")
    parser.add_argument("--batch", type=int, default=124, 
                        help="GSNN training batch size")
    parser.add_argument("--workers", type=int, default=10,
                        help="GSNN training; number of dataloader workers")
    
    # surrogate GSNN parameters
    parser.add_argument("--surr_channels", type=int, default=5,
                        help="GSNN hidden channels")
    parser.add_argument("--surr_layers", type=int, default=10,
                        help="GSNN number of layers")
    parser.add_argument("--surr_dropout", type=float, default=0.,
                        help="GSNN dropout probability")
    parser.add_argument("--surr_share_layers", action='store_true',
                        help="Whether GSNN should share parameters across layers")
    parser.add_argument("--surr_add_function_self_edges", action='store_true',
                        help="Whether GSNN should add self loops to the graph")
    parser.add_argument("--surr_norm", type=str, default='layer',
                        help="GSNN normalization method")
    parser.add_argument("--surr_bias", action='store_true',
                        help="GSNN should include a bias term")
    
    # surrogate GSNN training parameters
    parser.add_argument("--surr_lr", type=float, default=1e-2,
                        help="GSNN learning rate")
    parser.add_argument("--surr_epochs", type=int, default=500,
                        help="GSNN number of layers")
    parser.add_argument("--surr_patience", type=int, default=25,
                        help="early stopping patience")
    parser.add_argument("--surr_batch", type=int, default=3, 
                        help="GSNN training batch size")
    parser.add_argument("--surr_wd", type=float, default=0,
                        help="GSNN weight decay")
    
    # REINFORCE parameters 
    parser.add_argument("--rl_batch", type=int, default=20,
                        help="REINFORCE batch size")
    parser.add_argument("--rl_samples", type=int, default=50,
                        help="REINFORCE number of samples from surrogate environment")
    parser.add_argument("--rl_iters", type=int, default=250,
                        help="REINFORCE number of iterations to run")
    parser.add_argument("--rl_lr", type=float, default=1e-4,
                        help="REINFORCE learning rate")
    parser.add_argument("--rl_alpha", type=float, default=0.01,
                        help="REINFORCE policy initialization (smaller makes the candidate exploration more local)")
    
    # Surrogate GSNN hyperparameters 
    parser.add_argument("--stochastic_channels", type=int, default=4,
                        help="HyperNet stochastic channels")
    parser.add_argument("--hnet_width", type=int, default=8,
                        help="HyperNet number of channels")
    parser.add_argument("--samples", type=int, default=10,
                        help="HyperNet number of samples to draw during training")
    
    # bayesian optimization parameters
    parser.add_argument("--bayesopt_batch_size", type=int, default=2,
                        help="Number of candidate actions to eval each iteration")
    parser.add_argument("--record_dir", type=str, default='../ExpRec_tmp/',
                        help="directory path to save experiences to")
    parser.add_argument("--metric", type=str, default='spearman',
                        help="reward metric")
    parser.add_argument("--reward_agg", type=str, default='auc',
                        help="method of recording reward [last, best, auc]")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Number of warmup environment runs prior to starting bayesian optimization")
    parser.add_argument("--warmup_p", type=float, default=0.95,
                        help="probability of selecting actions during warmup; rough proportion of total actions taken during warmup")
    parser.add_argument("--iters", type=int, default=100,
                        help="number of bayesian optimization iterations to run")
    parser.add_argument("--acquisition", type=str, default='PI',
                        help="acquisition function to use for bayesopt exploration [LCB, UCB, EI, PI, mean]")
    parser.add_argument("--q", type=float, default=0.75,
                        help="LCB/UCB quantile value to use")
    parser.add_argument("--neighborhood", type=int, default=1000,
                        help="soft constraint on the neighborhood around the best reward that bayesopt should explore")

    # other params 
    parser.add_argument("--optimize", type=str, default='function_edges',
                        help="whether to optimize `input_edges` or `function_edges`")
    parser.add_argument("--add_false_edges", type=int, default=0,
                        help="Number of false edges to add to the graph")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save the results every X bayesian optimization iterations")
    
    args = parser.parse_args()

    return args


if __name__ == '__main__': 

    args = get_args()
    print(args)
    print()

    os.makedirs(args.out, exist_ok=True)
 
    siginfo = pd.read_csv(f'{args.siginfo}/siginfo_beta.txt', sep='\t', low_memory=False)

    data = torch.load(f'{args.data}/data.pt')

    print('# of false dti edges:', (~data.true_input_edge_mask).sum().item())

    if args.optimize == 'function_edges':
        key = ('function', 'to', 'function')
    elif args.optimize == 'input_edges':
        key = ('input', 'to', 'function')
    else:
        raise ValueError('optimize must be either `function` or `input`')

    train_ids = np.load(f'{args.fold}/lincs_train_obs.npy', allow_pickle=True)
    train_dataset = LincsDataset(root=f'{args.data}', sig_ids=train_ids, data=data, siginfo=siginfo)

    val_ids = np.load(f'{args.fold}/lincs_val_obs.npy', allow_pickle=True)
    val_dataset = LincsDataset(root=f'{args.data}', sig_ids=val_ids, data=data, siginfo=siginfo)

    bayesopt = BayesOpt(args, data, train_dataset, val_dataset, key=key)

    bayesopt.warmup(args.warmup, p=args.warmup_p)

    best_rewards = []
    for i in np.arange(args.iters//args.save_every): 

        best_reward_per_iter = bayesopt.run(iters=args.save_every, 
                                            objective=args.acquisition, 
                                            obj_kwargs={'q':args.q}, 
                                            neighborhood_size=args.neighborhood)
        
        best_rewards += best_reward_per_iter

        best_reward_idx = np.stack([np.array(r) for r in bayesopt.record.rewards], axis=0).mean(axis=(-1)).argmax()
        best_reward = bayesopt.record.rewards[best_reward_idx].mean()
        best_action = bayesopt.record.actions[best_reward_idx]

        torch.save({'bayesopt':bayesopt, 
                    'best_action':best_action, 
                    'best_reward':best_reward,
                    'best_rewards_per_iter': best_rewards,
                    'key':key,
                    'args':args,
                    'data':data}, 
                    args.out + '/bayesopt_results_dict.pt')
        
        if (~data.true_input_edge_mask).sum().item() > 0: 
            assert key == ('input', 'to', 'function'), 'using false dti edges should only be used with `input` optimization'

            src = data.edge_index_dict[key][0].detach().cpu().numpy()
            src_names = np.array(data.node_names_dict['input'])[src]
            dti_edge_mask = torch.tensor([True if 'DRUG_' in n else False for n in src_names], dtype=torch.bool)

            y = data.true_input_edge_mask[dti_edge_mask]
            yhat = best_action[dti_edge_mask]

            true_negs = ((y == 0) & (yhat == 0)).sum().item()
            tot_negs = (y == 0).sum().item()
            true_pos = ((y == 1) & (yhat==1)).sum().item()
            tot_pos = (y == 1).sum().item()
            
            print('best action dti edge accuracy (true vs. predicted):', (y == yhat).float().mean().item())
            print(f'true pos: {true_pos}/{tot_pos} [{true_pos/tot_pos:.2f}]| true negs: {true_negs}/{tot_negs} [{true_negs/tot_negs:.2f}]')
        





