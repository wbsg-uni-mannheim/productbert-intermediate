import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, calculate_prec_rec_f1, MetricTracker
from collections import OrderedDict
from model.loss import nll_loss, cross_entropy_loss


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, lr_scheduler, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = np.ceil(len(self.data_loader) / self.config['trainer']['accumulation_steps'])
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        # self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size * self.config['trainer']['accumulation_steps']))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.model.zero_grad()
        acc_loss, acc_output, acc_target = 0.0, None, None
        for batch_idx, (data, token_ids, attn_mask, target) in enumerate(self.data_loader):
            data, token_ids, attn_mask, target = data.to(self.device), token_ids.to(self.device), attn_mask.to(self.device), target.to(self.device)


            output = self.model(data, token_ids, attn_mask)

            if self.config.config.get('pos_neg_ratio'):
                loss = self.criterion(output, target, self.config['pos_neg_ratio'])
            else:
                loss = self.criterion(output, target)



            if self.config['trainer']['accumulation_steps'] > 1:
                loss = loss / self.config['trainer']['accumulation_steps']
            loss.backward()

            acc_loss += loss.item()
            if isinstance(acc_output, torch.Tensor) and isinstance(acc_target, torch.Tensor):
                acc_output = torch.cat([acc_output, output], dim=0)
                acc_target = torch.cat([acc_target, target], dim=0)
            else:
                acc_output = output
                acc_target = target

            if (batch_idx + 1) % self.config['trainer']['accumulation_steps'] == 0:

                global_batch_index = int(batch_idx / self.config['trainer']['accumulation_steps'])

                if self.config['clip_grad']:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad'])
                self.optimizer.step()

                if self.lr_scheduler is not None:
                    if self.config['lr_scheduler']['step_every_batch']:
                        if type(self.lr_scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau:
                            self.lr_scheduler.step(val_log['loss'])
                        else:
                            self.lr_scheduler.step()

                output = acc_output.sigmoid()

                self.model.zero_grad()


                self.writer.set_step((epoch - 1) * self.len_epoch + global_batch_index)
                self.train_metrics.update('loss', acc_loss)
                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(output, acc_target))



                if global_batch_index % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress(global_batch_index),
                        acc_loss))
                    #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                acc_loss, acc_output, acc_target = 0.0, None, None

                if global_batch_index == self.len_epoch:
                    break
        log = self.train_metrics.result()
        log_total = self.train_metrics.total()

        precision, recall, f1 = calculate_prec_rec_f1(log_total)

        if self.do_validation:
            val_log, val_log_total = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            val_precision, val_recall, val_f1 = calculate_prec_rec_f1(val_log_total)

        if self.lr_scheduler is not None:
            if not self.config['lr_scheduler']['step_every_batch']:
                if type(self.lr_scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau:
                    self.lr_scheduler.step(val_log['loss'])
                else:
                    self.lr_scheduler.step()


        if self.do_validation:
            additional_log = {"tp": log_total['tp'], "fp": log_total['fp'], "tn": log_total['tn'],
                              "fn": log_total['fn'], "precision": precision, "recall": recall, "f1": f1,
                              "val_tp": val_log_total['tp'], "val_fp": val_log_total['fp'],
                              "val_tn": val_log_total['tn'], "val_fn": val_log_total['fn'],
                              "val_precision": val_precision, "val_recall": val_recall, "val_f1": val_f1}
        else:
            additional_log = {"tp": log_total['tp'], "fp": log_total['fp'], "tn": log_total['tn'],
                              "fn": log_total['fn'], "precision": precision, "recall": recall, "f1": f1}
        log.update(additional_log)
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, token_ids, attn_mask, target) in enumerate(self.valid_data_loader):
                data, token_ids, attn_mask, target, = data.to(self.device), token_ids.to(self.device), attn_mask.to(self.device), target.to(self.device)

                output = self.model(data, token_ids, attn_mask)

                if self.config.config.get('pos_neg_ratio'):
                    loss = self.criterion(output, target, self.config['pos_neg_ratio'])
                else:
                    loss = self.criterion(output, target)

                output = output.sigmoid()

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result(), self.valid_metrics.total()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * (self.data_loader.batch_size * self.config['trainer']['accumulation_steps'])
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

class TrainerMLM(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, lr_scheduler, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = np.ceil(len(self.data_loader) / self.config['trainer']['accumulation_steps'])
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        # self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size * self.config['trainer']['accumulation_steps']))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.model.zero_grad()
        acc_loss, acc_output, acc_target = 0.0, None, None
        for batch_idx, (data, token_ids, attn_mask, target, mlm_target) in enumerate(self.data_loader):
            data, token_ids, attn_mask, target, mlm_target = data.to(self.device), token_ids.to(self.device), attn_mask.to(self.device), target.to(self.device), mlm_target.to(self.device)

            output = self.model(data, token_ids, attn_mask)
            mlm_output = output[1]
            output = output[0]

            if self.config.config.get('pos_neg_ratio'):
                loss_binary = self.criterion(output, target, self.config['pos_neg_ratio'])
            else:
                loss_binary = self.criterion(output, target)

            if self.config['n_gpu'] > 1:
                mlm_loss = cross_entropy_loss(mlm_output.view(-1, self.model.module.config.vocab_size), mlm_target.view(-1))
            else:
                mlm_loss = cross_entropy_loss(mlm_output.view(-1, self.model.config.vocab_size), mlm_target.view(-1))
            
            loss = loss_binary + mlm_loss

            if self.config['trainer']['accumulation_steps'] > 1:
                loss = loss / self.config['trainer']['accumulation_steps']
            loss.backward()

            acc_loss += loss.item()
            if isinstance(acc_output, torch.Tensor) and isinstance(acc_target, torch.Tensor):
                acc_output = torch.cat([acc_output, output], dim=0)
                acc_target = torch.cat([acc_target, target], dim=0)
            else:
                acc_output = output
                acc_target = target

            if (batch_idx + 1) % self.config['trainer']['accumulation_steps'] == 0:

                global_batch_index = int(batch_idx / self.config['trainer']['accumulation_steps'])

                if self.config['clip_grad']:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad'])
                self.optimizer.step()

                if self.lr_scheduler is not None:
                    if self.config['lr_scheduler']['step_every_batch']:
                        if type(self.lr_scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau:
                            self.lr_scheduler.step(val_log['loss'])
                        else:
                            self.lr_scheduler.step()

                output = acc_output.sigmoid()

                self.model.zero_grad()


                self.writer.set_step((epoch - 1) * self.len_epoch + global_batch_index)
                self.train_metrics.update('loss', acc_loss)
                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(output, acc_target))



                if global_batch_index % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress(global_batch_index),
                        acc_loss))
                    #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                acc_loss, acc_output, acc_target = 0.0, None, None

                if global_batch_index == self.len_epoch:
                    break
        log = self.train_metrics.result()
        log_total = self.train_metrics.total()

        precision, recall, f1 = calculate_prec_rec_f1(log_total)

        if self.do_validation:
            val_log, val_log_total = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            val_precision, val_recall, val_f1 = calculate_prec_rec_f1(val_log_total)

        if self.lr_scheduler is not None:
            if not self.config['lr_scheduler']['step_every_batch']:
                if type(self.lr_scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau:
                    self.lr_scheduler.step(val_log['loss'])
                else:
                    self.lr_scheduler.step()


        if self.do_validation:
            additional_log = {"tp": log_total['tp'], "fp": log_total['fp'], "tn": log_total['tn'],
                              "fn": log_total['fn'], "precision": precision, "recall": recall, "f1": f1,
                              "val_tp": val_log_total['tp'], "val_fp": val_log_total['fp'],
                              "val_tn": val_log_total['tn'], "val_fn": val_log_total['fn'],
                              "val_precision": val_precision, "val_recall": val_recall, "val_f1": val_f1}
        else:
            additional_log = {"tp": log_total['tp'], "fp": log_total['fp'], "tn": log_total['tn'],
                              "fn": log_total['fn'], "precision": precision, "recall": recall, "f1": f1}
        log.update(additional_log)
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, token_ids, attn_mask, target, mlm_labels) in enumerate(self.valid_data_loader):
                data, token_ids, attn_mask, target, mlm_labels = data.to(self.device), token_ids.to(self.device), attn_mask.to(self.device), target.to(self.device), mlm_labels.to(self.device)

                output = self.model(data, token_ids, attn_mask, mlm_labels)
                mlm_loss = output[1]
                output = output[0]


                if self.config.config.get('pos_neg_ratio'):
                    binary_loss = self.criterion(output, target, self.config['pos_neg_ratio'])
                else:
                    binary_loss = self.criterion(output, target)

                loss = binary_loss + mlm_loss

                output = output.sigmoid()

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result(), self.valid_metrics.total()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * (self.data_loader.batch_size * self.config['trainer']['accumulation_steps'])
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)