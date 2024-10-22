
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
    
    parser.add_argument("--warmup", type=int, default=10,
                        help="number of starting iters without gradient updates") 
    
    parser.add_argument("--action_channels", type=int, default=124,
                        help="the actor input feature dimension") 
    
    parser.add_argument("--env_repls", type=int, default=3,
                        help="number of times to run the environment per outer iter, the reward will be the average of rewards from replicates") 
    
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="decay factor to compute the exponential moving stats to normalize reward") 
    
    parser.add_argument("--inner_lr", type=float, default=1e-2,
                        help="GSNN learning rate") 
    
    parser.add_argument("--outer_lr", type=float, default=1e-3,
                        help="RL learning rate") 
    
    parser.add_argument("--l1_penalty", type=float, default=0,
                        help="This will penalize the bernoulli probabilities and encourage sparsity") 
    
    parser.add_argument("--init_entropy", type=float, default=0.1,
                        help="initial entropy coeficient; larger values will encourage exploration") 
    
    parser.add_argument("--entropy_decay", type=float, default=0.5,
                        help="entropy decay value") 
    
    parser.add_argument("--entropy_schedule", type=int, default=50,
                        help="number of iters per entropy value") 
    
    parser.add_argument("--min_entropy", type=float, default=0,
                        help="minimum entropy coeficient; larger values will encourage exploration") 
    
    parser.add_argument("--actor_bias", type=float, default=None,
                        help="initialization bias on probabilities (can help select more/less nodes at start)")
    
    parser.add_argument("--actor_model", type=str, default='nn',
                        help="actor model architecture [nn, linear, gat, gcn]")
    
    parser.add_argument("--actor_channels", type=int, default='32',
                        help="number of hidden channels to use in the actor model (not applicable to linear model)")
    
    parser.add_argument("--ppo_iters", type=int, default=1000,
                        help="number of policy updates per outer iteration")
    
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help="proximal policy clip parameter [0,1]")
    
    parser.add_argument("--target_kl", type=float, default=0.01,
                        help="proximal policy clip early stopping threshold")
    
    parser.add_argument("--checkpoint", action='store_true', default=False,
                        help="whether to use layerwise gradient checkpointing")
    
    parser.add_argument("--max_prop_actions", type=float, default=1.0,
                        help="selecting more than X percent of possible actions will result in penalty (-1); can be used to prevent selection of all actions")
    
    parser.add_argument("--selection", type=str, default='node',
                        help="whether to select nodes or edges [node, edge]")

    args = parser.parse_args()

    return args

def get_node_features(args, dgi_params={'epochs':100, 'dropout':0., 'layers':10, 'conv':'gin', 'lr':1e-4}, 
                                         n2v_params={'epochs':50, 'lr':1e-2}): 

    if args.features == 'reactome': 
        
        print('embedding genes using reactome pathway membership...')
        uni2rea = pd.read_csv(f'../../data/UniProt2Reactome_All_Levels.txt', sep='\t', header=None).rename({i:n for i,n in enumerate(['uniprot', 'pathway_id', 'url', 'name', '-', 'species'])}, axis=1)
        x, pathway_ids, pathway_names = get_function_pathway_features(data, uni2rea, K=1000)
        model = VAE(x, dropout=0., hidden_channels=512, latent_dim=args.action_channels)
        _,_ = model.train(device='cuda', epochs=1000, patience=-1, lr=1e-3, beta=1, verbose=True)
        z  = model.embed(device='cpu')
        x = torch.tensor(z, dtype=torch.float32)

    elif args.features == 'onehot': 

        x = torch.eye(len(data['node_names_dict']['function']), dtype=torch.float32)

    elif args.features == 'node2vec': 
        print('training node2vec model...')
        N2V = Node2Vec(data.edge_index_dict['function','to','function'], embedding_dim=args.action_channels, walk_length=20, context_size=10, 
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
        dgi = DGI(data_,  args.action_channels, dropout=dgi_params['dropout'], layers=dgi_params['layers'], conv=dgi_params['conv'])
        dgi.train(device, epochs=dgi_params['epochs'], lr=dgi_params['lr'])
        x = torch.tensor(dgi.embed(), dtype=torch.float32)
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)

    elif args.features == 'cat': 
        print('training node2vec model...')
        N2V = Node2Vec(data.edge_index_dict['function','to','function'], embedding_dim=args.action_channels//3, walk_length=20, context_size=10, 
                                    walks_per_node=25, num_negative_samples=1, p=1., q=0.5, sparse=True)
        N2V.train(epochs=n2v_params['epochs'], lr=n2v_params['lr'])
        x1 = torch.tensor(N2V.embed(), dtype=torch.float32)

        print('training deep graph infomax...')
        data_ = pyg.data.Data() 
        data_.edge_index = data['edge_index_dict']['function','to','function']
        data_.num_nodes = len(data['node_names_dict']['function'])
        data_.x = torch.eye(data_.num_nodes) # one hot encode nodes 
        dgi = DGI(data_, args.action_channels//3, dropout=dgi_params['dropout'], layers=dgi_params['layers'], conv=dgi_params['conv'])
        dgi.train(device, epochs=dgi_params['epochs'], lr=dgi_params['lr'])
        x2 = torch.tensor(dgi.embed(), dtype=torch.float32)

        print('embedding genes using reactome pathway membership...')
        uni2rea = pd.read_csv(f'../../data/UniProt2Reactome_All_Levels.txt', sep='\t', header=None).rename({i:n for i,n in enumerate(['uniprot', 'pathway_id', 'url', 'name', '-', 'species'])}, axis=1)
        x, pathway_ids, pathway_names = get_function_pathway_features(data, uni2rea, K=1000)
        model = VAE(x, dropout=0., hidden_channels=512, latent_dim=args.action_channels//3)
        _,_ = model.train(device='cuda', epochs=1000, patience=-1, lr=1e-3, beta=1, verbose=True)
        z  = model.embed(device='cpu')
        x3 = torch.tensor(z, dtype=torch.float32)

        # use both dgi and n2v features
        x = torch.cat((x1,x2,x3), dim=-1)
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
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

    x = get_node_features(args)

    if args.selection == 'edge':
        print('using "edge" selection, converting node embeddings to edge embeddings...')
        x = torch.stack([torch.cat((x[i.item()], x[j.item()]), dim=-1) for i,j in zip(*data.edge_index_dict['function', 'to', 'function'])], dim=0)

    actor = Actor(in_channels=x.size(1), bias=args.actor_bias, model=args.actor_model, hidden_channels=args.actor_channels).to(device)
    x = x.to(device)
    
    optim = torch.optim.Adam(actor.parameters(), lr=args.outer_lr)
    env = Environment(train_dataset, val_dataset, test_dataset, model_kwargs, training_kwargs, device, metric=args.metric)

    ppo = PPO(args, clip=3, eps=1e-3, warmup=3, verbose=True)

    if args.actor_model in ['gcn', 'gat']: 
        edge_index = data.edge_index_dict['function', 'to', 'function'].to(device)
        edge_index = to_undirected(edge_index)
        actor_ = lambda x: actor(x=x, edge_index=edge_index)
    elif args.actor_model in ['hgnn']: 

        edge_index_dict = {} 
        for k,v in data.edge_index_dict.items(): 
            if k == ('function', 'to', 'function'): 
                edge_index_dict[k] = to_undirected(v).to(device) 
            else: 
                edge_index_dict[k] = v.to(device) 
                edge_index_dict[(k[2], k[1], k[0])] = torch.stack((v[1], v[0]), dim=0).to(device)

        x_input = torch.tensor([1.*('DRUG__' in x) for x in data.node_names_dict['input']], dtype=torch.float32).to(device).view(-1,1)
        x_output = torch.ones((len(data.node_names_dict['output']),1), dtype=torch.float32).to(device)

        actor_ = lambda x: actor(x={'input'     :x_input,
                                    'function'  :x, 
                                    'output'    :x_output}, 
                                 edge_index=edge_index_dict)
    else: 
        actor_ = actor

    print()
    print()
    first=True
    for iter in range(args.outer_iters): 
        tic = time.time()
        optim.zero_grad()

        logits = actor_(x)
        # consider clipping logits during training to prevent fixing of certain nodes
        # need gradient, so only clip the logits that generate the action. 
        m = Bernoulli(logits=logits)

        if iter < args.warmup: 
            # open ai spinning up recommendation; sample from random dist during warmup
            action = 1.*(torch.rand_like(logits) > 0.5)
        else: 
            if first: 
                first = False 
                print()
                print('warmup over, beginning policy-based actions.')
            action = m.sample() 

        if action.mean() > args.max_prop_actions: 
            reward = -1
        else: 
            rewards = [] # rewards are multioutput (1, n_lincs)
            for _ in range(args.env_repls): 
                rewards.append( env.run(action, action_type=args.selection) )
            rewards = np.stack(rewards, axis=0)
            rewards = np.mean(rewards, axis=0)

        ppo.train_actor(logits, action, rewards, x, actor_, optim)
        ppo.update(rewards)

        with torch.no_grad(): 
            action_sum = action.sum() 
            print(f'\t\titer: {iter} || mean perf: {rewards.mean():.3f} || reward: {ppo.scale(ppo.rewards[-1]).mean():.3f} || # actions: {int(action_sum)}/{len(action)} || elapsed: {(time.time()-tic)/60:.2f} min')

        torch.save({'state_dict':actor.state_dict(), 
                    'node_prob':actor_(x).squeeze().sigmoid().detach().cpu().numpy(),
                    'ppo':ppo, 
                    'args':args,
                    'x':x, 
                    'actor':actor,
                    'data':data,
                    'model_kwargs':model_kwargs,
                    'training_kwargs':training_kwargs,
                    'env':env}, args.out + '/rl_results_dict.pt')






