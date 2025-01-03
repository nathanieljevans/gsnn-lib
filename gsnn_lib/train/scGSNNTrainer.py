from .Trainer import Trainer 
from sklearn.metrics import r2_score
import torch 
from geomloss import SamplesLoss    

# import wassertein loss 
from scipy.stats import wasserstein_distance_nd
from gsnn.external.mmd import compute_scalar_mmd 


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

        return yhat, y

    def _compute_metrics(self, y, yhat, loss, eval=False, **kwargs):
        """Compute and return metrics for the epoch."""

        if eval: # If we're evaluating on the epoch level (all data), we can compute more metrics.

            #mmd = compute_scalar_mmd(y, yhat)
            
            metrics = {
                'loss': loss,
                'neg_loss': -loss,
                #'mmd': mmd
            }

        else: # evaluating within batch (within condition in this case)

            with torch.no_grad():
                # R^2 (variance_weighted) as an example
                # NOTE: this is only relevant if it's being calculated *within* condition; so epoch level eval will be irrelevant 
                r2 = r2_score(y.mean(0), yhat.mean(0))

            with torch.no_grad(): 
                shd = SamplesLoss("sinkhorn", p=2, blur=0.05)(torch.tensor(y, dtype=torch.float32), torch.tensor(yhat, dtype=torch.float32)).item()
            wass = wasserstein_distance_nd(y, yhat)

            mmd = compute_scalar_mmd(y, yhat)

            metrics = {
                'loss': loss,
                'pop_mean_r2': r2,
                'wasserstein': wass,
                'sinkhorn': shd,
                'mmd': mmd
            }

        return metrics