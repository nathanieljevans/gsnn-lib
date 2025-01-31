import time
import numpy as np
import torch
from abc import ABC, abstractmethod
from sklearn.metrics import r2_score
import copy 


class Trainer(ABC):
    """
    A base Trainer class that defines a general interface for training,
    validation, and testing. Model-specific trainers like GSNNTrainer,
    NNTrainer, and GNNTrainer should inherit from this and implement
    their own domain-specific logic if necessary.
    """
    
    def __init__(self, model, optimizer, optim_params, criterion, device='cpu', 
                 logger=None, scheduler=None, early_stopper=None, prune_every=None,
                 prune_threshold=None):
        
        self.model = model
        self._optimizer = optimizer
        self._optim_params = optim_params
        self.optimizer = optimizer(model.parameters(), **optim_params)
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.scheduler = scheduler
        self.early_stopper = early_stopper
        self.best_model_state = None
        self.best_perf = -np.inf
        self.current_epoch = 0
        self.prune_every = prune_every
        self.prune_threshold = prune_threshold
        
        self.model.to(self.device)

    @abstractmethod
    def _run_batch(self, batch):
        """
        Defines how to handle a single batch. This must:
          1) Extract inputs (x) and targets (y).
          2) Move data to the appropriate device.
          3) Run a forward pass.
          4) Return the model predictions (yhat) and the targets (y).
        
        This method must be implemented by derived classes as it may vary
        depending on the model and data format.
        """
        pass

    def _step(self, batch):
        """
        Executes a single training step:
          1) Run a forward pass and get predictions.
          2) Compute the loss.
          3) Backpropagate and update parameters.
        
        Returns the loss and outputs for logging purposes.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        yhat, y, kwargs = self._run_batch(batch)
        loss = self.criterion(yhat, y)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), yhat, y, kwargs

    def _run_epoch(self, dataloader, training=True):
        """
        Runs one epoch of training or evaluation.
        For evaluation, `training=False` is set which avoids parameter updates.
        Returns metrics dictionary.
        """
        epoch_loss = []
        all_y = []
        all_yhat = []
        
        if training:
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(training):
            for i,batch in enumerate(dataloader):
                if training:
                    loss, yhat, y, kwargs = self._step(batch)
                else:
                    # Validation/Test step without backprop
                    yhat, y, kwargs = self._run_batch(batch)
                    loss = self.criterion(yhat, y).item()
                
                batch_metrics = self._compute_metrics(y.detach().cpu().numpy(), yhat.detach().cpu().numpy(), loss, kwargs)
                    
                epoch_loss.append(loss)
                all_y.append(y.detach().cpu().numpy())
                all_yhat.append(yhat.detach().cpu().numpy())

                # Print batch metrics
                metric_str = '...'.join([f'{k}: {v:.2f}' for k,v in batch_metrics.items()])
                print(f'[batch:{i+1}/{len(dataloader)}...{metric_str}', end='\r')

        # Convert predictions and targets
        all_y = np.concatenate(all_y, axis=0)
        all_yhat = np.concatenate(all_yhat, axis=0)
        
        # Compute metrics
        # Note `eval=True` computes (potentially) different metrics than eval=False 
        metrics = self._compute_metrics(all_y, all_yhat, np.mean(epoch_loss), {}, eval=True)
        return metrics, all_y, all_yhat

    @abstractmethod
    def _compute_metrics(self, y, yhat, loss, kwargs, eval=False):
        """
        Compute and return metrics given predictions and targets.
        Derived classes can implement custom metrics if needed.
        """
        pass

    def train_epoch(self, train_loader):
        metrics, _, _ = self._run_epoch(train_loader, training=True)
        
        if self.scheduler is not None:
            self.scheduler.step()

        return metrics

    def validate_epoch(self, val_loader):
        metrics, _, _ = self._run_epoch(val_loader, training=False)
        return metrics

    def test_epoch(self, test_loader):
        metrics, y, yhat = self._run_epoch(test_loader, training=False)
        return metrics, y, yhat

    def train(self, train_loader, val_loader, epochs=100, metric_key='r2'):
        """
        Full training loop with optional early stopping and logging.
        `metric_key` is the name of the metric to use for best model tracking.
        """
        for epoch in range(1, epochs+1):
            self.current_epoch = epoch
            start = time.time()

            torch.cuda.empty_cache()
            train_metrics = self.train_epoch(train_loader)
            torch.cuda.empty_cache()
            val_metrics = self.validate_epoch(val_loader)
            
            # Logging (if logger is available)
            if self.logger is not None:
                self.logger.log(epoch, train_metrics, val_metrics)

            # Track best performance and early stopping
            perf = val_metrics.get(metric_key, None)
            if perf is not None and perf > self.best_perf:
                self.best_perf = perf
                self.best_model_state = {k: v.cpu().detach().clone() for k, v in self.model.state_dict().items()}
            
            # Early stopping
            if self.early_stopper is not None and self.early_stopper.early_stop(-perf if perf is not None else None):
                print(f"Early stopping at epoch {epoch}")
                break

            elapsed = time.time() - start
            val_metric_str = '...'.join([f'{k}: {v:.3E}' for k,v in val_metrics.items()])
            print(f"Epoch {epoch}/{epochs}, Train Loss: {train_metrics['loss']:.4f} || -> Val -> || {val_metric_str} || Time Elapsed: {elapsed:.2f}s")

            if self.prune_every is not None: 
                if (epoch % self.prune_every == 0) and (epoch > 0):
                    print('-------------------------------------------------------------')
                    n_pruned = self.model.prune(self.prune_threshold, verbose=True)
                    n_params = sum(p.numel() for p in self.model.parameters())
                    print(f'# model parameters: {n_params} [# pruned: {n_pruned}]')
                    print('-------------------------------------------------------------')
                    # reinitialize optimizer
                    self.optimizer = self._optimizer(self.model.parameters(), **self._optim_params)

            
        # Load best model weights if available
        if self.best_model_state is not None:

            # hacky way to use state_dicts with GSNN + pruning 
            if self.prune_every is not None: self.match_lin_sizes_(self.model, self.best_model_state)

            self.model.load_state_dict(self.best_model_state)

    def test(self, test_loader):
        return self.test_epoch(test_loader)
    

    def match_lin_sizes_(self, model, state_dict): 
        '''
        if using iterative pruning, the number of parameters can change after pruning, leading to mismatch 
        in earlier/later model state_dict (from `best_model_state`) and model parameter dimensions. Only applies to GSNN.
        '''

        for i, mod in enumerate(self.model.ResBlocks):
            lin_in_size = state_dict[f'ResBlocks.{i}.lin_in.values'].size()
            lin_out_size = state_dict[f'ResBlocks.{i}.lin_out.values'].size()
            mod.lin_in.values = torch.nn.Parameter(torch.zeros(lin_in_size, dtype=torch.float32, device=self.device))
            mod.lin_out.values = torch.nn.Parameter(torch.zeros(lin_out_size, dtype=torch.float32, device=self.device))

            lin_in_ind_size = state_dict[f'ResBlocks.{i}.lin_in.indices'].size() 
            lin_out_ind_size = state_dict[f'ResBlocks.{i}.lin_out.indices'].size()
            mod.lin_in.register_buffer('indices', torch.zeros(lin_in_ind_size, dtype=torch.long, device=self.device))
            mod.lin_out.register_buffer('indices', torch.zeros(lin_out_ind_size, dtype=torch.long, device=self.device))