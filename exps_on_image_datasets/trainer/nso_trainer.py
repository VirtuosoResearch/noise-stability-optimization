from trainer.base_trainer import Trainer
from utils.bypass_bn import enable_running_stats, disable_running_stats

import torch
import torch.nn.functional as F
import os
from .base_trainer import Trainer
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn


class NSOTrainer(Trainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, 
                 train_data_loader, valid_data_loader=None, test_data_loader=None, 
                 lr_scheduler=None, checkpoint_dir=None,
                 num_perturbs = 1, nso_lam=0.5, use_neg=False):
        super().__init__(model, criterion, metric_ftns, optimizer, config, device, 
                         train_data_loader, valid_data_loader, test_data_loader, 
                         lr_scheduler, checkpoint_dir)
        self.num_perturbs = num_perturbs
        self.nso_lam = nso_lam
        self.use_neg = use_neg

        
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target, index) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            # first forward-backward step
            enable_running_stats(self.model)
            
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.store_gradients(zero_grad=True, store_weights=True, update_weight=self.nso_lam)

            # second forward-backward step
            disable_running_stats(self.model)
            if self.num_perturbs != 0:
                update_weight = (1-self.nso_lam)/(2*self.num_perturbs) if self.use_neg else (1-self.nso_lam)/(self.num_perturbs)
                for i in range(self.num_perturbs):
                    self.optimizer.first_step(zero_grad=True, store_perturb=True)
                    self.criterion(self.model(data), target).backward()
                    self.optimizer.store_gradients(zero_grad=True, store_weights=False, update_weight=update_weight)
                    if self.use_neg:
                        self.optimizer.first_step(zero_grad=True, store_perturb=False)
                        self.criterion(self.model(data), target).backward()
                        self.optimizer.store_gradients(zero_grad=True, store_weights=False, update_weight=update_weight)
            self.optimizer.second_step(zero_grad=True)

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log
    

class NSOSWATrainer(Trainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, 
                train_data_loader, valid_data_loader=None, test_data_loader=None, lr_scheduler=None, checkpoint_dir=None,
                num_perturbs = 1, nso_lam=0.5, use_neg=False, swa_start = 20, swa_lr = 0.0005):
        super().__init__(model, criterion, metric_ftns, optimizer, config, device, 
                train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir)
        self.num_perturbs = num_perturbs
        self.nso_lam = nso_lam
        self.use_neg = use_neg
        
        self.swa_model = AveragedModel(self.model)
        self.swa_start = swa_start
        self.swa_lr = swa_lr
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.swa_lr, anneal_epochs=5)
        self.save_epoch = 1

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target, index) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            # first forward-backward step
            enable_running_stats(self.model)
            
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.store_gradients(zero_grad=True, store_weights=True, update_weight=self.nso_lam)

            # second forward-backward step
            disable_running_stats(self.model)
            if self.num_perturbs != 0:
                update_weight = (1-self.nso_lam)/(2*self.num_perturbs) if self.use_neg else (1-self.nso_lam)/(self.num_perturbs)
                for i in range(self.num_perturbs):
                    self.optimizer.first_step(zero_grad=True, store_perturb=True)
                    self.criterion(self.model(data), target).backward()
                    self.optimizer.store_gradients(zero_grad=True, store_weights=False, update_weight=update_weight)
                    if self.use_neg:
                        self.optimizer.first_step(zero_grad=True, store_perturb=False)
                        self.criterion(self.model(data), target).backward()
                        self.optimizer.store_gradients(zero_grad=True, store_weights=False, update_weight=update_weight)
            self.optimizer.second_step(zero_grad=True)

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if epoch >= self.swa_start:
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
        else:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        return log
    
    def test(self):
        update_bn(self.train_data_loader, self.swa_model, device=self.device)

        self.swa_model.eval()

        total_loss = 0.0
        total_metrics = torch.zeros(len(self.metric_ftns))

        with torch.no_grad():
            for i, batch in enumerate(self.test_data_loader):
                if len(batch) == 3:
                    data, target, index = batch
                else:
                    data, target = batch
                data, target = data.to(self.device), target.to(self.device)
                output = self.swa_model(data)

                # computing loss, metrics on test set
                loss = self.criterion(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(self.metric_ftns):
                    total_metrics[i] += metric(output, target) * batch_size

        n_samples = len(self.test_data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(self.metric_ftns)
        })
        self.logger.info(log)

        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': self.epochs,
            'state_dict': self.swa_model.state_dict(),
        }
        
        best_path = os.path.join(self.checkpoint_dir, "swa_model.pth")
        torch.save(state, best_path)
        self.logger.info("Saving weight averaging model: swa_model.pth ...")

        return log[self.metric_ftns[0].__name__]