
import torch 
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data import DataLoader
import pandas as pd 
import argparse 
import numpy as np
import time 
import torch_geometric as pyg 

from gsnn.reinforce.Actor import Actor
from gsnn.reinforce.Environment import Environment
from gsnn.reinforce.RewardScaler import RewardScaler
from gsnn.models.GSNN import GSNN
from gsnn.data.LincsDataset import LincsDataset
from gsnn.models import utils 
from gsnn.proc.utils import get_x_drug_conc           # required to unpickle data 
from gsnn.reinforce.Node2Vec import Node2Vec
from gsnn.reinforce.DGI import DGI
from gsnn.reinforce import ppo
from gsnn.proc.utils import get_function_pathway_features
from gsnn.reinforce.VAE import VAE 


def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../processed_data/',
                        help="path to data directory")

    parser.add_argument("--fold", type=str, default='../processed_data/',
                        help="path to data fold directory; must contain data splits - see `create_data_splits.py`")
    
    parser.add_argument("--out", type=str, default='../processed_data/',
                        help="output directory")
    
    parser.add_argument("--siginfo", type=str, default='../../data/',
                        help="path to siginfo directory")
    
    parser.add_argument("--batch", type=int, default=100,
                        help="training batch size")
    
    parser.add_argument("--workers", type=int, default=10,
                        help="number of workers to use for dataloaders")

    parser.add_argument("--channels", type=int, default=3,
                        help="number of GSNN hidden channels")        

    parser.add_argument("--layers", type=int, default=10,
                        help="number of GSNN layers")   
    
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout probability (regularization)")  
    
    parser.add_argument("--norm", type=str, default='none',
                        help="normalization layer [none, layer]")
    
    parser.add_argument("--init", type=str, default='kaiming',
                        help="initialization style [kaiming, xavier]")
    
    parser.add_argument("--metric", type=str, default='spearman',
                        help="metric to optimize in reinforcement learning [pearson, spearman, mse, r2]")
    
    parser.add_argument("--features", type=str, default='dgi',
                        help="node features to use for predicting node selection [reactome, node2vec, dgi]")
    
    parser.add_argument("--inner_iters", type=int, default=25,
                        help="number of epochs to train the gsnn model")   
    
    parser.add_argument("--outer_iters", type=int, default=1000,
                        help="number of iters to train the RL model")  

    parser.add_argument("--patience", type=int, default=5,
                        help="number of epochs without val improvement before stopping") 
    
    parser.add_argument("--alpha", type=float, default=0.04,
                        help="decay factor to compute the exponential moving stats to normalize reward") 
    
    parser.add_argument("--inner_lr", type=float, default=1e-4,
                        help="GSNN learning rate") 
    
    parser.add_argument("--outer_lr", type=float, default=1e-4,
                        help="RL learning rate") 
    
    parser.add_argument("--l1_penalty", type=float, default=0,
                        help="This will penalize the bernoulli probabilities and encourage sparsity") 
    
    parser.add_argument("--entropy", type=float, default=1e-2,
                        help="entropy coeficient; larger values will encourage exploration") 
    
    parser.add_argument("--actor_bias", type=float, default=None,
                        help="initialization bias on probabilities (can help select more/less nodes at start)")
    
    parser.add_argument("--actor_model", type=str, default='nn',
                        help="actor model architecture [nn, linear]")
    
    parser.add_argument("--alg", type=str, default='ppo',
                        help="type of RL algorithm to use [reinforce, ppo]")
    
    parser.add_argument("--ppo_iters", type=int, default=4,
                        help="number of policy updates per outer iteration")
    
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help="proximal policy clip parameter [0,1]")
    
    parser.add_argument("--checkpoint", action='store_true', default=False,
                        help="whether to use layerwise gradient checkpointing")
    
    parser.add_argument("--max_prop_actions", type=float, default=0.8,
                        help="selecting more than X percent of possible actions will result in penalty (-1); can be used to prevent selection of all actions")
    
    args = parser.parse_args()

    return args

def get_node_features(args, channels=124, dgi_params={'epochs':100, 'dropout':0., 'layers':10, 'conv':'gin', 'lr':1e-4}, 
                                         n2v_params={'epochs':50, 'lr':1e-2}): 

    if args.features == 'reactome': 
        
        print('embedding genes using reactome pathway membership...')
        uni2rea = pd.read_csv(f'../../data/UniProt2Reactome_All_Levels.txt', sep='\t', header=None).rename({i:n for i,n in enumerate(['uniprot', 'pathway_id', 'url', 'name', '-', 'species'])}, axis=1)
        x, pathway_ids, pathway_names = get_function_pathway_features(data, uni2rea, K=1000)
        model = VAE(x, dropout=0., hidden_channels=512, latent_dim=channels)
        _,_ = model.train(device='cuda', epochs=1000, patience=-1, lr=1e-3, beta=1, verbose=True)
        z  = model.embed(device='cpu')
        x = torch.tensor(z, dtype=torch.float32)

    elif args.features == 'onehot': 

        x = torch.eye(len(data['node_names_dict']['function']), dtype=torch.float32)

    elif args.features == 'node2vec': 
        print('training node2vec model...')
        N2V = Node2Vec(data.edge_index_dict['function','to','function'], embedding_dim=channels, walk_length=20, context_size=10, 
                                    walks_per_node=100, num_negative_samples=1, p=1., q=0.5, sparse=True)
        N2V.train(epochs=n2v_params['epochs'], lr=n2v_params['lr'])
        x = torch.tensor(N2V.embed(), dtype=torch.float32)
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)

    elif args.features == 'dgi': 
        print('training deep graph infomax...')
        data_ = pyg.data.Data() 
        data_.edge_index = data['edge_index_dict']['function','to','function']
        data_.num_nodes = len(data['node_names_dict']['function'])
        data_.x = torch.eye(data_.num_nodes) # one hot encode nodes 
        dgi = DGI(data_,  channels, dropout=dgi_params['dropout'], layers=dgi_params['layers'], conv=dgi_params['conv'])
        dgi.train(device, epochs=dgi_params['epochs'], lr=dgi_params['lr'])
        x = torch.tensor(dgi.embed(), dtype=torch.float32)
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)

    elif args.features == 'cat': 
        print('training node2vec model...')
        N2V = Node2Vec(data.edge_index_dict['function','to','function'], embedding_dim=channels//2, walk_length=20, context_size=10, 
                                    walks_per_node=25, num_negative_samples=1, p=1., q=0.5, sparse=True)
        N2V.train(epochs=n2v_params['epochs'], lr=n2v_params['lr'])
        x1 = torch.tensor(N2V.embed(), dtype=torch.float32)
        x1 = (x1 - x1.mean(dim=0)) / (x1.std(dim=0) + 1e-8)

        print('training deep graph infomax...')
        data_ = pyg.data.Data() 
        data_.edge_index = data['edge_index_dict']['function','to','function']
        data_.num_nodes = len(data['node_names_dict']['function'])
        data_.x = torch.eye(data_.num_nodes) # one hot encode nodes 
        dgi = DGI(data_, channels//2, dropout=dgi_params['dropout'], layers=dgi_params['layers'], conv=dgi_params['conv'])
        dgi.train(device, epochs=dgi_params['epochs'], lr=dgi_params['lr'])
        x2 = torch.tensor(dgi.embed(), dtype=torch.float32)
        x2 = (x2 - x2.mean(dim=0)) / (x2.std(dim=0) + 1e-8)

        print('embedding genes using reactome pathway membership...')
        uni2rea = pd.read_csv(f'../../data/UniProt2Reactome_All_Levels.txt', sep='\t', header=None).rename({i:n for i,n in enumerate(['uniprot', 'pathway_id', 'url', 'name', '-', 'species'])}, axis=1)
        x, pathway_ids, pathway_names = get_function_pathway_features(data, uni2rea, K=1000)
        model = VAE(x, dropout=0., hidden_channels=512, latent_dim=channels)
        _,_ = model.train(device='cuda', epochs=1000, patience=-1, lr=1e-3, beta=1, verbose=True)
        z  = model.embed(device='cpu')
        x3 = torch.tensor(z, dtype=torch.float32)

        # use both dgi and n2v features
        x = torch.cat((x1,x2,x3), dim=-1)
    else:
        raise Exception()
    
    return x 




if __name__ == '__main__': 

    args = get_args()

    if torch.cuda.is_available(): 
        device = 'cuda'
    else: 
        device = 'cpu'
    print('using device:', device)

    siginfo = pd.read_csv(f'{args.siginfo}/siginfo_beta.txt', sep='\t', low_memory=False)

    data = torch.load(f'{args.data}/data.pt')

    train_ids = np.load(f'{args.fold}/lincs_train_obs.npy', allow_pickle=True)
    train_dataset = LincsDataset(root=f'{args.data}', sig_ids=train_ids, data=data, siginfo=siginfo)

    val_ids = np.load(f'{args.fold}/lincs_val_obs.npy', allow_pickle=True)
    val_dataset = LincsDataset(root=f'{args.data}', sig_ids=val_ids, data=data, siginfo=siginfo)

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
                       'min_delta':1e-2,
                       'batch':args.batch,
                       'workers':args.workers}

    x = get_node_features(args)

    actor = Actor(in_channels=x.size(1), bias=args.actor_bias, model=args.actor_model)
    scaler = RewardScaler(alpha=args.alpha, clip=5)
    
    optim = torch.optim.Adam(actor.parameters(), lr=args.outer_lr)
    env = Environment(train_dataset, val_dataset, model_kwargs, training_kwargs, device, metric=args.metric)

    print()
    print()
    for iter in range(args.outer_iters): 
        tic = time.time()
        optim.zero_grad()

        logits = actor(x).squeeze()
        m = Bernoulli(logits=logits)
        action = m.sample() 

        if action.mean() > args.max_prop_actions: 
            reward = -1
        else: 
            reward = env.run(action)

        reward_scaled = scaler.scale(reward)

        # REINFORCE
        if args.alg == 'reinforce': 
            loss = -reward_scaled*m.log_prob(action).sum() + args.l1_penalty*logits.mean() + args.entropy*m.entropy().mean()
            loss.backward()
            optim.step() 
        elif args.alg == 'ppo': 
            ppo.update_actor(args, actor, x, reward_scaled, logits, action, optim, iters=args.ppo_iters, clip_param=args.clip_param)
        else: 
            raise Exception 
        
        scaler.update(reward)

        with torch.no_grad(): 
            reward_mean, reward_std = scaler.get_params()
            action_sum = action.sum() 
            print(f'\t\titer: {iter} || perf: {reward:.3f} || reward: {reward_scaled:.3f} || # actions: {int(action_sum)}/{len(action)} || reward stats (mean, std): ({reward_mean:.2f},{reward_std:.2f}) || elapsed: {(time.time()-tic)/60:.2f} min')

        torch.save({'state_dict':actor.state_dict(), 
                    'node_prob':actor(x).squeeze().sigmoid().detach().cpu().numpy(),
                    'reward_scaler':scaler, 
                    'args':args,
                    'x':x, 
                    'actor':actor,
                    'data':data,
                    'model_kwargs':model_kwargs,
                    'training_kwargs':training_kwargs,
                    'env':env}, args.out + '/rl_results_dict.pt')






