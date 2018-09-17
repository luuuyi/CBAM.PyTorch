import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time
import sys
import os

class Trainer():
    def __init__(self, model, model_type, loss_fn, optimizer, lr_schedule, log_batchs, is_use_cuda, train_data_loader, \
                valid_data_loader=None, metric=None, start_epoch=0, num_epochs=25, is_debug=False, logger=None, writer=None):
        self.model = model
        self.model_type = model_type
        self.loss_fn  = loss_fn
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.log_batchs = log_batchs
        self.is_use_cuda = is_use_cuda
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.metric = metric
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.is_debug = is_debug

        self.cur_epoch = start_epoch
        self.best_acc = 0.
        self.best_loss = sys.float_info.max
        self.logger = logger
        self.writer = writer

    def fit(self):
        for epoch in range(0, self.start_epoch):
            self.lr_schedule.step()

        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.append('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            self.logger.append('-' * 60)
            self.cur_epoch = epoch
            self.lr_schedule.step()
            if self.is_debug:
                self._dump_infos()
            self._train()
            self._valid()
            self._save_best_model()
            print()

    def _dump_infos(self):
        self.logger.append('---------------------Current Parameters---------------------')
        self.logger.append('is use GPU: ' + ('True' if self.is_use_cuda else 'False'))
        self.logger.append('lr: %f' % (self.lr_schedule.get_lr()[0]))
        self.logger.append('model_type: %s' % (self.model_type))
        self.logger.append('current epoch: %d' % (self.cur_epoch))
        self.logger.append('best accuracy: %f' % (self.best_acc))
        self.logger.append('best loss: %f' % (self.best_loss))
        self.logger.append('------------------------------------------------------------')

    def _train(self):
        self.model.train()  # Set model to training mode
        losses = []
        if self.metric is not None:
            self.metric[0].reset()

        for i, (inputs, labels) in enumerate(self.train_data_loader):              # Notice
            if self.is_use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
                labels = labels.squeeze()
            else:
                labels = labels.squeeze()

            self.optimizer.zero_grad()
            outputs = self.model(inputs)            # Notice 
            loss = self.loss_fn[0](outputs, labels)
            if self.metric is not None:
                prob     = F.softmax(outputs, dim=1).data.cpu()
                self.metric[0].add(prob, labels.data.cpu())
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())       # Notice
            if 0 == i % self.log_batchs or (i == len(self.train_data_loader) - 1):
                local_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                batch_mean_loss  = np.mean(losses)
                print_str = '[%s]\tTraining Batch[%d/%d]\t Class Loss: %.4f\t'           \
                            % (local_time_str, i, len(self.train_data_loader) - 1, batch_mean_loss)
                if i == len(self.train_data_loader) - 1 and self.metric is not None:
                    top1_acc_score = self.metric[0].value()[0]
                    top5_acc_score = self.metric[0].value()[1]
                    print_str += '@Top-1 Score: %.4f\t' % (top1_acc_score)
                    print_str += '@Top-5 Score: %.4f\t' % (top5_acc_score)
                self.logger.append(print_str)
        self.writer.add_scalar('loss/loss_c', batch_mean_loss, self.cur_epoch)

    def _valid(self):
        self.model.eval()
        losses = []
        acc_rate = 0.
        if self.metric is not None:
            self.metric[0].reset()

        with torch.no_grad():              # Notice
            for i, (inputs, labels) in enumerate(self.valid_data_loader):
                if self.is_use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    labels = labels.squeeze()
                else:
                    labels = labels.squeeze()

                outputs = self.model(inputs)            # Notice 
                loss = self.loss_fn[0](outputs, labels)

                if self.metric is not None:
                    prob     = F.softmax(outputs, dim=1).data.cpu()
                    self.metric[0].add(prob, labels.data.cpu())
                losses.append(loss.item())
            
        local_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        #self.logger.append(losses)
        batch_mean_loss = np.mean(losses)
        print_str = '[%s]\tValidation: \t Class Loss: %.4f\t'     \
                    % (local_time_str, batch_mean_loss)
        if self.metric is not None:
            top1_acc_score = self.metric[0].value()[0]
            top5_acc_score = self.metric[0].value()[1]
            print_str += '@Top-1 Score: %.4f\t' % (top1_acc_score)
            print_str += '@Top-5 Score: %.4f\t' % (top5_acc_score)
        self.logger.append(print_str)
        if top1_acc_score >= self.best_acc:
            self.best_acc = top1_acc_score
            self.best_loss = batch_mean_loss

    def _save_best_model(self):
        # Save Model
        self.logger.append('Saving Model...')
        state = {
            'state_dict': self.model.state_dict(),
            'best_acc': self.best_acc,
            'cur_epoch': self.cur_epoch,
            'num_epochs': self.num_epochs
        }
        if not os.path.isdir('./checkpoint/' + self.model_type):
            os.makedirs('./checkpoint/' + self.model_type)
        torch.save(state, './checkpoint/' + self.model_type + '/Models' + '_epoch_%d' % self.cur_epoch + '.ckpt')   # Notice