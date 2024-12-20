
import os
import numpy as np
import pandas as pd
import torch
import sklearn
import scipy

def agg_fold_metrics(path):

    folds = np.sort([x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))])
    metrics = []
    for fold in folds:
        fold_path = os.path.join(path, fold)
        metrics.append(torch.load(os.path.join(fold_path, 'result_metric_dict.pt'), weights_only=False))

    avg_metrics = {} 
    for key in metrics[0].keys():
        avg_metrics[key] = np.mean([x[key] for x in metrics])

    return avg_metrics
    

def agg_fold_predictions(path): 
    '''
    Aggregate test predictions from all folds in a given path (uid)
    '''

    folds = np.sort([x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))])
    
    preds = [] 
    sig_ids = []
    for fold in folds:
        pred = torch.load(path + '/' + fold + '/test_predictions.pt', weights_only=False)
        sids = torch.load(path + '/' + fold + '/test_sig_ids.pt', weights_only=False)
        # BUG: ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (131,) + inhomogeneous part.
        # only occurs for GSNN (not GNN or NN)
        if pred.shape[0] != len(sids): sids = np.concatenate(sids, 0)
        preds.append( pred )
        sig_ids.append( sids ) 

    sig_ids = np.concatenate(sig_ids, 0)
    preds = np.concatenate(preds, 0)

    return preds, sig_ids

def load_y(proc, sig_ids): 
    ''' 
    Load y from processed data
    '''

    y = [] 
    for i,sig_id in enumerate(sig_ids): 
        print('progress: ', i, ' / ', len(sig_ids), end='\r')
        y.append( torch.load(proc + '/obs/' + sig_id + '.pt', weights_only=True) ) 

    y = torch.stack(y, 0)
    y = y.detach().cpu().numpy()
    return y 

def grouped_eval(y, preds, cond_ids, data, condinfo):

    condinfo = condinfo[lambda x: x.condition_id.isin(cond_ids)]  
    sig2idx = {x:i for i,x in enumerate(cond_ids)}

    drug_metrics = {'pert_id': [], 'r2': [], 'mse': [], 'pearsonr': [], 'spearmanr':[], 'N':[]}
    for i,row in condinfo[['pert_id', 'condition_id']].groupby('pert_id').agg(lambda x: list(x)).reset_index().iterrows(): 
        print('calculating drug-grouped metrics... ', i, ' / ', len(condinfo.pert_id.unique()), end='\r')
        didxs = [sig2idx[x] for x in row.condition_id]
        yy = y[didxs]
        yyhat = preds[didxs] 

        drug_metrics['pert_id'].append(row.pert_id)
        drug_metrics['r2'].append(sklearn.metrics.r2_score(yy, yyhat, multioutput='variance_weighted'))
        drug_metrics['mse'].append( np.mean((yy - yyhat)**2) )  
        drug_metrics['pearsonr'].append( np.mean([np.corrcoef(yy[i], yyhat[i])[0,1] for i in range(yy.shape[0])]) )
        drug_metrics['spearmanr'].append( np.mean([scipy.stats.spearmanr(yy[i], yyhat[i]).statistic for i in range(yy.shape[0])]) )
        drug_metrics['N'].append(yy.shape[0])

    drug_metrics = pd.DataFrame(drug_metrics)
    print()

    ### performance by cell line 
    cell_metrics = {'cell_iname': [], 'r2': [], 'mse': [], 'pearsonr': [], 'spearmanr':[], 'N':[]}
    for i,row in condinfo[['cell_iname', 'condition_id']].groupby('cell_iname').agg(lambda x: list(x)).reset_index().iterrows(): 
        print('calculating cell-grouped metrics... ', i, ' / ', len(condinfo.cell_iname.unique()), end='\r')
        didxs = [sig2idx[x] for x in row.condition_id]
        yy = y[didxs]
        yyhat = preds[didxs] 

        cell_metrics['cell_iname'].append(row.cell_iname)
        cell_metrics['r2'].append(sklearn.metrics.r2_score(yy, yyhat, multioutput='variance_weighted'))
        cell_metrics['mse'].append( np.mean((yy - yyhat)**2) )  
        cell_metrics['pearsonr'].append( np.mean([np.corrcoef(yy[i], yyhat[i])[0,1] for i in range(yy.shape[0])]) )
        cell_metrics['spearmanr'].append( np.mean([scipy.stats.spearmanr(yy[i], yyhat[i]).statistic for i in range(yy.shape[0])]) )
        cell_metrics['N'].append(yy.shape[0])

    cell_metrics = pd.DataFrame(cell_metrics)
    print()

    ### performance by genes
    lincspace = data['node_names_dict']['output']
    gene_metrics = {'gene': [], 'r2': [], 'mse': [], 'pearsonr': [], 'spearmanr': []}
    for i in range(y.shape[1]): 
        print('calculating gene-grouped metrics... ', i, ' / ', y.shape[1], end='\r')

        yy = y[:,i]
        yyhat = preds[:,i]
        gene = lincspace[i].split('__')[1]

        gene_metrics['gene'].append(gene)
        gene_metrics['r2'].append(sklearn.metrics.r2_score(yy, yyhat))
        gene_metrics['mse'].append(np.mean((yy - yyhat)**2))
        gene_metrics['pearsonr'].append(np.corrcoef(yy, yyhat)[0,1])
        gene_metrics['spearmanr'].append(scipy.stats.spearmanr(yy, yyhat).statistic)
            
    gene_metrics = pd.DataFrame(gene_metrics)
    print() 

    return drug_metrics, cell_metrics, gene_metrics
