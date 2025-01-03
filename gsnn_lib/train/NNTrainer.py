import torch
import numpy as np
from sklearn.metrics import r2_score
from .Trainer import Trainer  # Adjust the path as needed

class NNTrainer(Trainer):
    """
    NNTrainer is a specialized trainer for NN models.
    It implements _run_batch and _compute_metrics methods for an NN model.
    """

    def _run_batch(self, batch):
        """
        Processes a single batch for the NN model.

        Batch structure: (x, y, sig_id)
        """
        x, y, sig_id = batch
        x = x.to(self.device)
        y = y.to(self.device)

        yhat = self.model(x)
        return yhat, y

    def _compute_metrics(self, y, yhat, loss, eval=False):
        """
        Compute metrics given numpy arrays of ground truths (y) and predictions (yhat).
        Returns a dict of metrics: loss, r2, r_flat.
        """
        r2 = r2_score(y, yhat, multioutput='variance_weighted')
        r_flat = np.corrcoef(y.ravel(), yhat.ravel())[0,1]
        return {
            'loss': loss,
            'r2': r2,
            'r_flat': r_flat
        }
