# -*- coding: utf-8 -*-
# @Time        : 2021/1/14 22:55
# @Author      : ssxy00
# @File        : trainer.py
# @Description :

import os
import logging
import math
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter


class MultiStepSchedule:
    def __init__(self, total_steps, gamma, init_lr, optimizer, milestones=(4/7, 6/7)):
        self.total_steps = total_steps
        self.milestone_steps = [int(total_steps * milestone) for milestone in milestones]
        self.init_lr = init_lr
        self.lr = init_lr
        self.gamma = gamma
        self.optimizer = optimizer
        self._step = 0

    def state_dict(self):
        return {'step': self._step,
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self._step = state_dict['step']
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def zero_grad(self):
        return self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        if step in self.milestone_steps:
            self.lr = self.lr * self.gamma
        return self.lr

class FRTrainer:
    def __init__(self, train_dataset, valid_dataset, model, args):
        # log
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__file__)
        self.writer = SummaryWriter(args.log_dir)

        # config
        self.device = torch.device(args.device)
        self.args = args

        # data
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        # model
        self.model = model
        if args.last_ckpt:
            last_ckpt_path = os.path.join(args.save_model_dir, f"checkpoint{args.last_ckpt}.pt")
            assert os.path.exists(last_ckpt_path), f"{last_ckpt_path} does not exist!"
            logging.info(f"Loading checkpoint from {last_ckpt_path}")
            self.model.load_state_dict(torch.load(last_ckpt_path))
        self.model.to(self.device)

        # optimize
        self.ce_criterion = nn.CrossEntropyLoss()
        # base_optimizer = SGD(self.model.parameters(), lr=args.lr)
        # total_steps = math.ceil(self.args.n_epochs * len(self.train_dataset) / self.args.batch_size)
        # self.optimizer = MultiStepSchedule(total_steps=total_steps, gamma=0.1, init_lr=args.lr,
        #                                    optimizer=base_optimizer)
        self.optimizer = SGD(self.model.parameters(), lr=args.lr)


    def train_epoch(self, epoch):
        self.model.train()
        n_total_samples = 0
        n_correct_samples = 0
        avg_ce_loss = 0.

        train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=8)
        tqdm_data = tqdm(train_dataloader, desc='Train (epoch #{})'.format(epoch))
        for i, data in enumerate(tqdm_data):
            data = {key: data[key].to(self.device) for key in data}
            logits = self.model(data["images"])
            # loss
            batch_ce_loss = self.ce_criterion(logits, data['labels'])
            batch_loss = batch_ce_loss  # TODO center loss
            batch_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            avg_ce_loss = (avg_ce_loss * i + batch_ce_loss.item()) / (i + 1)
            n_correct_samples += sum(logits.argmax(-1) == data["labels"]).item()
            n_total_samples += logits.shape[0]
            tqdm_data.set_postfix({'batch_ce_loss': batch_ce_loss.item(),
                                   'ave_ce_loss': avg_ce_loss,
                                   'acc': n_correct_samples / n_total_samples,
                                   'lr': self.optimizer.param_groups[0]['lr']})

            self.writer.add_scalar('Train/ce_loss', batch_ce_loss, (epoch - 1) * len(tqdm_data) + i + 1)
            self.writer.add_scalar('Train/avg_ce_loss', avg_ce_loss, (epoch - 1) * len(tqdm_data) + i + 1)
            self.writer.add_scalar('Train/acc', n_correct_samples / n_total_samples,
                                   (epoch - 1) * len(tqdm_data) + i + 1)
            self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'],
                                   (epoch - 1) * len(tqdm_data) + i + 1)





    def valid_epoch(self, epoch):
        self.model.eval()
        n_total_samples = 0
        n_correct_samples = 0
        avg_ce_loss = 0.
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=8)
        tqdm_data = tqdm(valid_dataloader, desc='Valid (epoch #{})'.format(epoch))
        for i, data in enumerate(tqdm_data):
            data = {key: data[key].to(self.device) for key in data}
            logits = self.model(data["images"])
            # loss
            batch_ce_loss = self.ce_criterion(logits, data['labels'])
            avg_ce_loss = (avg_ce_loss * i + batch_ce_loss.item()) / (i + 1)
            n_correct_samples += sum(logits.argmax(-1) == data["labels"]).item()
            n_total_samples += logits.shape[0]
            tqdm_data.set_postfix({'batch_ce_loss': batch_ce_loss.item(),
                                   'ave_ce_loss': avg_ce_loss,
                                   'acc': n_correct_samples / n_total_samples})

        self.writer.add_scalar('Valid/avg_ce_loss', avg_ce_loss, epoch)
        self.writer.add_scalar('Valid/acc', n_correct_samples / n_total_samples, epoch)



    def train(self):
        self.logger.info("begin to train")
        for epoch_idx in range(self.args.last_ckpt, self.args.n_epochs + 1):
            self.train_epoch(epoch_idx)
            self.valid_epoch(epoch_idx)
            if epoch_idx % self.args.save_interval == 0:
                save_path = os.path.join(self.args.save_model_dir, f"checkpoint{epoch_idx}.pt")
                torch.save(self.model.state_dict(), save_path)



