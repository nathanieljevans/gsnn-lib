
import torch 
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data import DataLoader
import pandas as pd 
import argparse 
import numpy as np
import time 
import torch_geometric as pyg 
from sklearn.metrics import roc_auc_score

from gsnn.reinforce.Actor import GSNNActor
from gsnn.reinforce.Actor import EmbeddedActor
from gsnn.reinforce.Environment import Environment
from gsnn.models.GSNN import GSNN
from gsnn.data.LincsDataset import LincsDataset
from gsnn.models import utils 
from gsnn.proc.utils import get_x_drug_conc           # required to unpickle data 
from gsnn.reinforce.Node2Vec import Node2Vec
from gsnn.reinforce.DGI import DGI
from gsnn.reinforce.PPO import PPO
from gsnn.proc.utils import get_function_pathway_features
from gsnn.reinforce.VAE import VAE 
from torch_geometric.utils import to_undirected 

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
    
    parser.add_argument("--batch", type=int, default=100,
                        help="training batch size")
    
    parser.add_argument("--workers", type=int, default=10,
                        help="number of workers to use for dataloaders")

    parser.add_argument("--channels", type=int, default=5,
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
    
    parser.add_argument("--inner_iters", type=int, default=25,
                        help="number of epochs to train the gsnn model")   
    
    parser.add_argument("--outer_iters", type=int, default=1000,
                        help="number of iters to train the RL model")  

    parser.add_argument("--patience", type=int, default=5,
                        help="number of epochs without val improvement before stopping") 
    
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="decay factor to compute the exponential moving stats to normalize reward") 
    
    parser.add_argument("--inner_lr", type=float, default=1e-2,
                        help="GSNN learning rate") 
    
    parser.add_argument("--outer_lr", type=float, default=1e-3,
                        help="RL learning rate") 
    
    parser.add_argument("--init_entropy", type=float, default=0.1,
                        help="initial entropy coeficient; larger values will encourage exploration") 
    
    parser.add_argument("--entropy_decay", type=float, default=0.5,
                        help="entropy decay value") 
    
    parser.add_argument("--entropy_schedule", type=int, default=50,
                        help="number of iters per entropy value") 
    
    parser.add_argument("--min_entropy", type=float, default=0,
                        help="minimum entropy coeficient; larger values will encourage exploration") 
    
    parser.add_argument("--ppo_iters", type=int, default=1000,
                        help="number of policy updates per outer iteration")
    
    parser.add_argument("--ppo_batch", type=int, default=3,
                        help="number of actions/env-runs to perform every iteration")
    
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help="proximal policy clip parameter [0,1]")
    
    parser.add_argument("--target_kl", type=float, default=0.01,
                        help="proximal policy clip early stopping threshold")
    
    parser.add_argument("--add_false_edges", type=int, default=0,
                        help="the number of False (random) edges")
    
    parser.add_argument("--checkpoint", action='store_true', default=False,
                        help="whether to use layerwise gradient checkpointing")
    
    args = parser.parse_args()

    return args


if __name__ == '__main__': 

    args = get_args()

    if torch.cuda.is_available(): 
        device = 'cuda'
    else: 
        device = 'cpu'
    print('using device:', device)

    siginfo = pd.read_csv(f'{args.siginfo}/siginfo_beta.txt', sep='\t', low_memory=False)

    data = torch.load(f'{args.data}/data.pt')

    if args.add_false_edges > 0: 
        data.edge_index_dict['function', 'to', 'function'], true_edge_mask = augment_edge_index(data.edge_index_dict['function', 'to', 'function'], N=args.add_false_edges)
    else: 
        true_edge_mask = None

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
                       'patience':args.patience,
                       'min_delta':1e-3,
                       'batch':args.batch,
                       'workers':args.workers}

    actor = EmbeddedActor(num_actions = data.edge_index_dict['function', 'to', 'function'].size(1), embed_dim=10)
    #actor = GSNNActor(data.edge_index_dict, data.node_names_dict, channels=2, layers=args.layers)
    x = torch.zeros((1,)) # dummy for now 
    
    optim = torch.optim.Adam(actor.parameters(), lr=args.outer_lr)
    env = Environment(train_dataset, val_dataset, test_dataset, model_kwargs, training_kwargs, device, metric=args.metric)

    ppo = PPO(args, clip=5, eps=1e-3, warmup=2, verbose=True)

    best_reward = -np.inf
    best_action = None

    print()
    print()
    for iter in range(args.outer_iters): 
        tic = time.time()
        optim.zero_grad()

        # compute a batch 
        logits = actor(x).squeeze()
        _actions = [] ; _rewards = []
        for ii in range(args.ppo_batch):
            #print(f'[progress: {ii+1}/{args.ppo_batch}]', end='\r')
            m = Bernoulli(logits=logits)
            action = m.sample() 

            rewards = env.run(action, action_type='edge', reward_type='auc', verbose=True)

            if rewards.mean() > best_reward: 
                best_reward = rewards.mean()
                best_action = action.detach().cpu().numpy()

            _actions.append(action) ; _rewards.append(rewards)

        ppo.train_actor(logits, _actions, _rewards, x, actor, optim)
        for r in _rewards: ppo.update(r)

        with torch.no_grad(): 
            action_sum = np.mean([a.sum() for a in _actions])
            batch_reward_mean = np.stack(_rewards).mean()
            if true_edge_mask is not None: 
                auroc = roc_auc_score((~true_edge_mask).detach().cpu().numpy(), 1-logits.sigmoid().detach().cpu().numpy())
            else: 
                auroc = -666
            print(f'\t\titer: {iter} || mean perf: {batch_reward_mean:.3f} || edge auroc: {auroc:.3f} || # actions: {int(action_sum)}/{len(action)} || elapsed: {(time.time()-tic)/60:.2f} min')

        torch.save({'state_dict':actor.state_dict(), 
                    'node_prob':actor(x).squeeze().sigmoid().detach().cpu().numpy(),
                    'ppo':ppo, 
                    'args':args,
                    'true_edge_mask':true_edge_mask,
                    'x':x, 
                    'actor':actor,
                    'best_reward':best_reward,
                    'best_action':best_action,
                    'data':data,
                    'model_kwargs':model_kwargs,
                    'training_kwargs':training_kwargs,
                    'env':env}, args.out + '/rl_results_dict.pt')






