import torch
import numpy as np
from sklearn.metrics import r2_score
from .Trainer import Trainer  # Assuming Trainer is in trainer.py or adjust import accordingly

class GSNNTrainer(Trainer):
    """
    GSNNTrainer is a specialized trainer for GSNN models.
    It inherits from Trainer and implements methods specific 
    to running batches and computing metrics for the GSNN model.
    """
    
    def _run_batch(self, batch):
        """
        Process a single batch.
        
        Args:
            batch: A tuple (x, y, sig_id)
                  x: input features (torch.Tensor)
                  y: target outputs (torch.Tensor)
                  sig_id: additional identifiers (not used in forward)
        
        Returns:
            yhat: Model predictions
            y: Ground truth targets
        """
        x, y, sig_id = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Forward pass
        yhat = self.model(x)
        return yhat, y, {}

    def _compute_metrics(self, y, yhat, loss, kwargs, eval=False):
        """
        Compute metrics given numpy arrays of ground truths and predictions.
        
        Args:
            y (np.ndarray): Ground truth targets.
            yhat (np.ndarray): Model predictions.
            loss (float): The mean loss over the epoch.
        
        Returns:
            dict: A dictionary of computed metrics.
        """
        
        # Compute R^2 score. Using variance_weighted as per original code
        r2 = r2_score(y, yhat, multioutput='variance_weighted')
        
        # Compute flat correlation
        r_flat = np.corrcoef(y.ravel(), yhat.ravel())[0, 1]

        return {
            'loss': loss,
            'r2': r2,
            'r_flat': r_flat
        }
