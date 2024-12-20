
import torch 
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data import DataLoader
import pandas as pd 
import argparse 
import numpy as np
import time 
import torch_geometric as pyg 
from sklearn.metrics import roc_auc_score

from gsnn.optim.Environment import Environment
from gsnn.models.GSNN import GSNN
from gsnn_lib.data.LincsDataset import LincsDataset
from gsnn.models import utils 
from gsnn_lib.proc.lincs.utils import get_x_drug_conc           # required to unpickle data 

from gsnn_lib.utils.augment_edge_index import augment_edge_index

from gsnn.optim.REINFORCE import REINFORCE
from gsnn.optim.BayesOpt import BayesOptAgent
import os 

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
    
    parser.add_argument("--batch", type=int, default=64,
                        help="training batch size")
    
    parser.add_argument("--workers", type=int, default=10,
                        help="number of workers to use for dataloaders")

    parser.add_argument("--channels", type=int, default=10,
                        help="number of GSNN hidden channels")        

    parser.add_argument("--layers", type=int, default=10,
                        help="number of GSNN layers")   
    
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout probability (regularization)")  
    
    parser.add_argument("--norm", type=str, default='layer',
                        help="normalization layer [none, layer]")
    
    parser.add_argument("--init", type=str, default='kaiming',
                        help="initialization style [kaiming, xavier]")
    
    parser.add_argument("--metric", type=str, default='spearman',
                        help="metric to optimize in reinforcement learning [pearson, spearman, mse, r2]")
    
    parser.add_argument("--reward_type", type=str, default='auc',
                        help="reward type to use in reinforcement learning [auc, best, last]")
    
    parser.add_argument("--inner_iters", type=int, default=25,
                        help="number of epochs to train the gsnn model")   
    
    parser.add_argument("--outer_iters", type=int, default=1000,
                        help="number of iters to train the RL model")  

    parser.add_argument("--save_every", type=int, default=10,
                        help="saves model results and weights every X epochs")
    
    parser.add_argument("--window", type=int, default=10,
                        help="window size for reward averaging")
    
    parser.add_argument("--inner_lr", type=float, default=1e-2,
                        help="GSNN learning rate") 
    
    parser.add_argument("--outer_lr", type=float, default=1e-1,
                        help="RL learning rate") 
    
    parser.add_argument("--init_entropy", type=float, default=0.,
                        help="initial entropy coeficient; larger values will encourage exploration") 
    
    parser.add_argument("--entropy_decay", type=float, default=0.9,
                        help="entropy decay value") 
    
    parser.add_argument("--min_entropy", type=float, default=0,
                        help="minimum entropy coeficient; larger values will encourage exploration") 
    
    parser.add_argument("--checkpoint", action='store_true', default=False,
                        help="whether to use layerwise gradient checkpointing")
    
    parser.add_argument("--init_probs", type=float, default=0.5,
                        help="initial probability of sampling an edge")
    
    parser.add_argument("--policy_decay", type=float, default=0.,
                        help="decay factor for policy entropy")
    
    parser.add_argument("--warmup", type=int, default=5,
                        help="number of iterations to warmup the policy")
    
    parser.add_argument("--eps", type=float, default=1e-6,
                        help="epsilon value for numerical stability")
    
    parser.add_argument("--reward_clip", type=float, default=10,
                        help="reward clipping value")
    
    parser.add_argument("--method", type=str, default='reinforce',
                        help="optimization method [reinforce, bayesopt]")
    
    args = parser.parse_args()

    return args


if __name__ == '__main__': 

    args = get_args()
    print(args)
    print() 

    os.makedirs(args.out, exist_ok=True)

    if torch.cuda.is_available(): 
        device = 'cuda'
    else: 
        device = 'cpu'
    print('using device:', device)

    siginfo = pd.read_csv(f'{args.siginfo}/siginfo_beta.txt', sep='\t', low_memory=False)

    data = torch.load(f'{args.data}/data.pt')

    print('# of false dti edges:', (~data.true_input_edge_mask).sum().item())

    train_ids = np.load(f'{args.fold}/lincs_train_obs.npy', allow_pickle=True)
    train_dataset = LincsDataset(root=f'{args.data}', sig_ids=train_ids, data=data, siginfo=siginfo)

    val_ids = np.load(f'{args.fold}/lincs_val_obs.npy', allow_pickle=True)
    val_dataset = LincsDataset(root=f'{args.data}', sig_ids=val_ids, data=data, siginfo=siginfo)

    test_ids = np.load(f'{args.fold}/lincs_test_obs.npy', allow_pickle=True)
    test_dataset = LincsDataset(root=f'{args.data}', sig_ids=val_ids, data=data, siginfo=siginfo)

    model_kwargs = {'edge_index_dict'                 : data.edge_index_dict, 
                    'node_names_dict'                 : data.node_names_dict,
                    'channels'                        : args.channels, 
                    'layers'                          : args.layers, 
                    'dropout'                         : args.dropout,
                    'residual'                        : True,
                    'nonlin'                          : torch.nn.GELU,
                    'bias'                            : True,
                    'share_layers'                    : True,
                    'fix_hidden_channels'             : True,
                    'two_layer_conv'                  : False, 
                    'add_function_self_edges'         : True,
                    'norm'                            : args.norm,
                    'init'                            : args.init,
                    'checkpoint'                      : args.checkpoint}
    
    training_kwargs = {'lr':args.inner_lr, 
                       'max_epochs':args.inner_iters, 
                       'batch':args.batch,
                       'workers':args.workers}

    
    non_drug_node_idxs = torch.tensor([i for i,n in enumerate(data.node_names_dict['input']) if 'DRUG_' not in n], dtype=torch.long)
    non_drug_edge_mask = torch.tensor([s in non_drug_node_idxs for i,(s,t) in enumerate(zip(*data.edge_index_dict['input', 'to', 'function']))], dtype=torch.bool)
    drug_edge_mask = ~non_drug_edge_mask
    n_actions = drug_edge_mask.sum().item()
    action_idxs = -1*torch.ones((len(non_drug_edge_mask),), dtype=torch.long)
    action_idxs[drug_edge_mask] = torch.arange(n_actions, dtype=torch.long)
    action_edge_dict = {('input', 'to', 'function'): action_idxs}
    true_action = data.true_input_edge_mask[drug_edge_mask]

    env = Environment(action_edge_dict, train_dataset, test_dataset, model_kwargs, 
                    training_kwargs, metric=args.metric, reward_type=args.reward_type, verbose=True,
                    raise_error_on_fail=True)

    if args.method == 'reinforce': 
        hoptim = REINFORCE(env, n_actions, action_labels=true_action.detach().cpu().numpy(), clip=args.reward_clip, eps=args.eps, 
                           warmup=args.warmup, verbose=True,entropy=args.init_entropy, entropy_decay=args.entropy_decay, 
                           min_entropy=args.min_entropy, window=args.window, init_prob=args.init_probs, lr=args.outer_lr, 
                           policy_decay=args.policy_decay)
    elif args.method == 'bayesopt':
        hoptim = BayesOptAgent(env, n_actions, action_labels=true_action, warmup=args.warmup, verbose=False, suppress_warnings=True)
    else:
        raise ValueError('method must be either "reinforce" or "bayesopt"')
    
    print()
    print()
    for iter in range(args.outer_iters): 
        tic = time.time()

        hoptim.step()

        if (iter % args.save_every) == 0:
            
            torch.save({'hoptim':hoptim, 
                        'action_edge_dict':action_edge_dict,
                        'true_action':true_action,
                        'args':args,
                        'best_reward':hoptim.best_reward,
                        'best_action':hoptim.best_action,
                        'data':data,
                        'model_kwargs':model_kwargs,
                        'training_kwargs':training_kwargs}, args.out + '/dti_optim_results_dict.pt')






