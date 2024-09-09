
import torch 
import pandas as pd 
from sklearn.metrics import r2_score
import numpy as np 
import torch.nn.functional as F


class MutualInformationLoss(torch.nn.Module):
    def __init__(self, bins=30, max_val=None, min_val=None):
        super(MutualInformationLoss, self).__init__()
        self.bins = bins
        self.max_val = max_val
        self.min_val = min_val

    def forward(self, x, y):
        # Compute histograms
        if self.max_val is not None and self.min_val is not None:
            joint_hist = torch.histc(torch.stack([x, y], dim=-1), bins=self.bins, min=self.min_val, max=self.max_val)
            hist_x = torch.histc(x, bins=self.bins, min=self.min_val, max=self.max_val)
            hist_y = torch.histc(y, bins=self.bins, min=self.min_val, max=self.max_val)
        else:
            joint_hist = torch.histc(torch.stack([x, y], dim=-1), bins=self.bins)
            hist_x = torch.histc(x, bins=self.bins)
            hist_y = torch.histc(y, bins=self.bins)

        # Normalize histograms to get probability distributions
        joint_prob = joint_hist / joint_hist.sum()
        prob_x = hist_x / hist_x.sum()
        prob_y = hist_y / hist_y.sum()

        # Compute mutual information
        MI = torch.sum(joint_prob * (torch.log(joint_prob) - torch.log(prob_x.unsqueeze(-1)) - torch.log(prob_y.unsqueeze(-2))))

        # Return negative MI as loss (since we want to maximize MI)
        return -MI


class PearsonCorrelationLoss(torch.nn.Module):
    def __init__(self):
        super(PearsonCorrelationLoss, self).__init__()

    def forward(self, x, y):
        # Mean values
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)

        # Numerator: covariance of x and y
        cov_xy = torch.mean((x - mean_x) * (y - mean_y))

        # Denominator: standard deviations of x and y
        std_x = torch.sqrt(torch.mean((x - mean_x) ** 2) + 1e-8)  # Adding a small value to avoid division by zero
        std_y = torch.sqrt(torch.mean((y - mean_y) ** 2) + 1e-8)

        # Pearson Correlation Coefficient
        pcc = cov_xy / (std_x * std_y)

        # Return negative PCC as loss
        return -pcc

class GSNNExplainer: 

    def __init__(self, model, data, ignore_cuda=False, gumbel_softmax=True, hard=False, tau0=3, min_tau=0.5, prior=1, normalize_target=True): 
        '''
        Adapted from the methods presented in `GNNExplainer` (https://arxiv.org/abs/1903.03894). 

        Args: 
            model           torch.nn.Module             GSNN Model 
            data            pyg.Data                    GSNN processed graph data
            beta            float                       regularization scalar encouraging a minimal subset of edges
            ignore_cuda     bool                        whether to use cuda if available 
            hard            bool                        discrete forward operation for gumbel-softmax 
            tau0            float                       initial temperature value for gumbel-softmax 
            min_tau         float                       minimum temperature value for gumbel-softmax 
            prior           float                       prior strength to initialize theta; value of 0 will make each element 0.5 prob of being selecting, value > 0 will make it more likely to be selected. 

        Returns 
            None 
        '''
        self.gumbel_softmax = gumbel_softmax
        self.normalize_target = normalize_target
        self.prior = prior
        self.hard = hard
        self.min_tau = min_tau
        self.tau0 = tau0
        self.data = data
        self.device = 'cuda' if (torch.cuda.is_available() and not ignore_cuda) else 'cpu'
        
        model = model.eval()
        model = model.to(self.device)
        for p in model.parameters(): 
            p.requires_grad = False 
        self.model = model

    def explain(self, x, targets=None, iters=250, lr=1e-2, weight_decay=1e-5, 
                      beta=1, verbose=True, optimizer=torch.optim.SGD, desired_edges=1000, eps=1e-8): 
        '''
        initializes and runs gradient descent to select a minimal subset of edges that produce comparable predictions to the full graph. 
        
        Args: 
            x               torch.tensor            inputs to explain; in shape (B, I, 1)
            targets         listlike<int>           targets to explain, must be smaller than output dimensions 
            iters           int                     number of training iterations to perform 
            lr              float                   optimzer learning rate 
            weight_decay    float                   optimization weight decay on parameters 
            beta            float                   regularization, encourages minimal subset of edges 
            verbose         bool                    whether to print training progress 
            desired_edges   int                     desired number of edges subgraph will have 

        Returns 
            dataframe                               edgelist with importance scores 
        '''
        
        

        if x.size(0) > 1: 
            raise NotImplementedError('GSNNExplainer currently only supports single observation explanations')

        x = x.to(self.device).detach()

        # get baseline prediction "yb"
        target = self.model(x).detach()

        if self.normalize_target: 
            _mean = target.mean() 
            _std = target.std() 
            print('z-scoring target || (target dist params) mean, std:', (_mean, _std))
            target = (target - _mean)/(_std + eps)

        
        if self.gumbel_softmax: 
            # initialize parameter mask 
            weights = torch.stack((self.prior*torch.ones(self.model.edge_index.size(1), dtype=torch.float32, device=self.device, requires_grad=True), 
                                -self.prior*torch.ones(self.model.edge_index.size(1), dtype=torch.float32, device=self.device, requires_grad=True)), dim=0)
            edge_params = torch.nn.Parameter(weights)
        else: 
            # continuous weights 
            edge_params = torch.nn.Parameter(self.prior + torch.zeros(self.model.edge_index.size(1), dtype=torch.float32, device=self.device, requires_grad=True))

        # optimize parameter mask with objective 
        crit = torch.nn.MSELoss()   # MutualInformationLoss() #PearsonCorrelationLoss() #
        optim = optimizer([edge_params], lr=lr, weight_decay=weight_decay)

        # calculate tau decay rate
        tau_decay_rate = (self.min_tau / self.tau0) ** (1 / iters)

        if targets is None: targets = torch.arange(len(x.shape[1]))

        batch = x.size(0)

        for iter in range(iters):    
            optim.zero_grad()

            tau = max(self.tau0 * tau_decay_rate**iter, self.min_tau)

            mask = []
            for b in range(batch):
                if self.gumbel_softmax:
                    _mask, _rev_mask = torch.nn.functional.gumbel_softmax(edge_params, dim=0, hard=self.hard, tau=tau)
                    mask.append(_mask.view(1, -1, 1))
                else: 
                    mask.append(torch.sigmoid(edge_params).view(1, -1, 1))

            mask = torch.cat(mask, dim=0)

            out = self.model(x.expand(batch, -1, -1), mask=mask)  

            if self.normalize_target: 
                out = (out - _mean)/(_std +  eps)

            _target = target.expand(batch, -1)[:, targets].detach()
            _target_recon = out[:, targets]
            
            reg_loss = beta*mask.sum()/batch #(max(mask.sum()/batch, desired_edges) - desired_edges)

            mse = crit(_target_recon, _target)
            loss = mse + reg_loss
            loss.backward() 
            optim.step() 

            # avg explained variance
            
            if targets is not None: 
                if len(targets) == 1: 
                    r2 = -666 
                else: 
                    r2 = np.mean([r2_score(target[0, targets].detach().cpu().numpy(), out[i, targets].detach().cpu().numpy()) for i in range(out.size(0))])
            else: 
                r2 = np.mean([r2_score(target[0, :].detach().cpu().numpy(), out[i, :].detach().cpu().numpy()) for i in range(out.size(0))])

            if verbose: print(f'iter: {iter} || loss: {mse:.3f} || r2: {r2:.2f} || reg_loss: {reg_loss:.2f} || tau: {tau:.2f} || avg num sel edges: {mask.sum()/batch:.1f}', end='\r')

        if self.gumbel_softmax: 
            edge_scores, _ = torch.nn.functional.softmax(edge_params.data, dim=0).detach().cpu().numpy()
        else: 
            edge_scores = torch.sigmoid(edge_params).detach().cpu().numpy()
        src,dst = self.data.node_names[self.model.edge_index.detach().cpu().numpy()]
        edgedf = pd.DataFrame({'source':src, 'target':dst, 'score':edge_scores})
        return edgedf