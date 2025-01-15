from .Trainer import Trainer 
from sklearn.metrics import r2_score
import torch 
from geomloss import SamplesLoss    

# import wassertein loss 
from scipy.stats import wasserstein_distance_nd
from gsnn.external.mmd import compute_scalar_mmd 
import numpy as np 

class scGSNNTrainer(Trainer):
    """
    A specialized trainer for single-cell GSNN. Inherits from the base Trainer.
    """
    def _run_batch(self, batch):
        """
        In the scSamplerIterator, each batch is (X, y, x_cell, x_drug, y0).
        We'll move them to device, run forward pass, and return (yhat, y).
        """
        X, y, y0 = batch

        X = X.to(self.device)
        y = y.to(self.device)
        y0 = y0.to(self.device)

        # Forward pass:
        yhat = self.model(X) + y0  # example usage

        kwargs = {'y0': y0}

        return yhat, y, kwargs

    def _compute_metrics(self, y, yhat, loss, kwargs, eval=False):
        """Compute and return metrics for the epoch."""

        if eval: # If we're evaluating on the epoch level (all data), we can compute more metrics.

            #mmd = compute_scalar_mmd(y, yhat)
            
            metrics = {
                'loss': loss,
                'neg_loss': -loss,
            }


        else: # evaluating within batch (within condition in this case)

            with torch.no_grad():
                # R^2 (variance_weighted) as an example
                # NOTE: this is only relevant if it's being calculated *within* condition; so epoch level eval will be irrelevant 
                r2 = r2_score(y.mean(0), yhat.mean(0))

                y0_mean = kwargs['y0'].detach().cpu().numpy().mean(0)
                r2_delta = r2_score(y.mean(0) - y0_mean, yhat.mean(0) - y0_mean)

            with torch.no_grad(): 
                shd = SamplesLoss("sinkhorn", p=2, blur=0.05)(torch.tensor(y, dtype=torch.float32), torch.tensor(yhat, dtype=torch.float32)).item()
            wass = wasserstein_distance_nd(y, yhat)

            mmd = compute_scalar_mmd(y, yhat)

            metrics = {
                'loss': loss,
                'pop_mean_r2': r2,
                'pop_delta_mean_r2': r2_delta,
                'wasserstein': wass,
                'sinkhorn': shd,
                'mmd': mmd
            }

        return metrics