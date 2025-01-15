import torch
import numpy as np
from sklearn.metrics import r2_score
from .Trainer import Trainer  # Assuming you have the Trainer base class in .trainer

class GNNTrainer(Trainer):
    """
    GNNTrainer is a specialized trainer for GNN models.
    It implements _run_batch and _compute_metrics methods
    for GNN data using PyG datasets.
    """

    def _run_batch(self, batch):
        """
        Processes a single batch for the GNN model.
        
        Batch is expected to have:
          - batch.edge_index
          - batch.x
          - batch.y
          - batch.output_node_mask (boolean mask for output nodes)
          - batch.sig_id for identifying the samples
        """
        edge_index_dict = {k:v.to(self.device) for k,v in batch.edge_index_dict.items()}
        
        x_dict = {key: x.to(self.device) for key, x in batch.x_dict.items()}
        yhat_dict = self.model(x_dict,edge_index_dict)
        
        # Select output nodes
        yhat = yhat_dict['output']
        y = batch.y_dict['output'].to(self.device)

        return yhat, y, {}

    def _compute_metrics(self, y, yhat, loss, kwargs, eval=False):
        r2 = r2_score(y, yhat, multioutput='variance_weighted')
        r_flat = np.corrcoef(y.ravel(), yhat.ravel())[0,1]

        return {
            'loss': loss,
            'r2': r2,
            'r_flat': r_flat
        }
