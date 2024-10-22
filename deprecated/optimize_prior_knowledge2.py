
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
from gsnn.reinforce.PPO import PPO

from gsnn.reinforce.ExperienceRecord import ExperienceRecord
from gsnn.reinforce.SurrogateEnvironment import SurrogateEnvironment

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
    
    parser.add_argument("--ppo_iters", type=int, default=1000,
                        help="number of policy updates per outer iteration")
    
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help="proximal policy clip parameter [0,1]")
    
    parser.add_argument("--target_kl", type=float, default=0.01,
                        help="proximal policy clip early stopping threshold")
    
    parser.add_argument("--warmup", type=int, default=10,
                        help="initial warmup period") 

    parser.add_argument("--patience", type=int, default=5,
                        help="number of epochs without val improvement before stopping") 
    
    parser.add_argument("--min_surr_obs", type=int, default=100,
                        help="minimum number of observations to begin using surrogate environment") 
    
    parser.add_argument("--train_surr_every", type=int, default=25,
                        help="minimum number of observations to begin using surrogate environment") 
    
    parser.add_argument("--surr_batch_size", type=int, default=100,
                        help="the number of actions to evaluate in the surrogate environment every iter") 
    
    parser.add_argument("--surr_n_models", type=int, default=10,
                        help="the number of actions to evaluate in the surrogate environment every iter") 
    
    parser.add_argument("--inner_lr", type=float, default=1e-2,
                        help="GSNN learning rate") 
    
    parser.add_argument("--outer_lr", type=float, default=1e-3,
                        help="GSNN learning rate") 
    
    parser.add_argument("--init_entropy", type=float, default=1e-1,
                        help="initial entropy coefficient") 
    
    parser.add_argument("--entropy_decay", type=float, default=0.5,
                        help="entropy decay value") 
    
    parser.add_argument("--entropy_schedule", type=int, default=50,
                        help="number of iters per entropy value") 
    
    parser.add_argument("--min_entropy", type=float, default=0,
                        help="minimum entropy coeficient; larger values will encourage exploration") 
    
    parser.add_argument("--checkpoint", action='store_true', default=False,
                        help="whether to use layerwise gradient checkpointing")
    
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="decay factor to compute the exponential moving stats to normalize reward") 
    
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

    x = torch.eye(len(data['node_names_dict']['function']), dtype=torch.float32)

    actor = Actor(in_channels=x.size(1), model='linear')
    optim = torch.optim.Adam(actor.parameters(), lr=args.outer_lr)
    ppo = PPO(args, clip=3, eps=1e-3, warmup=3, verbose=True)
    env = Environment(train_dataset, val_dataset, test_dataset, model_kwargs, training_kwargs, device, metric=args.metric)
    venv = SurrogateEnvironment(data, n_models=25)
    record = ExperienceRecord(args.out)

    print()
    print()
    use_venv = False
    first = True
    for iter in range(args.outer_iters): 

        if (len(record) > args.min_surr_obs) & (((iter - args.warmup) % args.train_surr_every) == 0) & (iter >= args.warmup): 
            venv.optimize(record)
            use_venv = True

        tic = time.time()

        optim.zero_grad()

        logits = actor(x).view(1,-1)
        if use_venv: 
            vm = Bernoulli(logits=logits.expand(args.surr_batch_size, -1))
            vactions = vm.sample() # (surr_batch, num nodes)
            vactions, vrewards = venv.run(vactions, d=ppo.get_reward_params()[0])

            if vactions is None: 
                print(); print('no confident surrogate predictions')
            else: 
                print(); print(f'num confident predictions: {len(vactions)}/{args.surr_batch_size}')
                vrewards = vrewards #/ len(vactions)
        else:
            vactions = None; vrewards=None

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

        reward = env.run(action.squeeze(), action_type='node')
        record.add(action, reward)
        reward_scaled = ppo.scale(reward)
        if vrewards is not None: 
            rewards_ = np.array(vrewards.tolist() + [reward_scaled])
            actions_ = torch.cat((vactions, action.view(1,-1)), dim=0).contiguous()
        else: 
            rewards_ = np.array([reward_scaled])
            actions_ = action.view(1,-1)

        ppo.train_actor(logits, actions_, rewards_, x, actor, optim)
        ppo.update(reward) # update baseline

        with torch.no_grad(): 
            reward_mean, reward_std = ppo.get_reward_params()
            action_sum = action.sum() 
            print(f'\t\titer: {iter} || perf: {reward:.3f} || reward: {ppo.scale(ppo.rewards[-1]):.3f} || # actions: {int(action_sum)}/{len(action)} || reward stats (mean, std): ({reward_mean:.2f},{reward_std:.2f}) || elapsed: {(time.time()-tic)/60:.2f} min')

        torch.save({'state_dict':actor.state_dict(), 
                    'node_prob':actor(x).squeeze().sigmoid().detach().cpu().numpy(),
                    'ppo':ppo, 
                    'args':args,
                    'x':x, 
                    'actor':actor,
                    'data':data,
                    'model_kwargs':model_kwargs,
                    'training_kwargs':training_kwargs,
                    'env':env}, args.out + '/rl_results_dict.pt')

        
