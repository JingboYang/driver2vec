import sys
sys.path.append('.')

import time
import pickle

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn

from constants import *


class Optimizer():

    def __init__(self, model_params, dataset_len, args):

        self.args = args
        self.train_args = args.train_args
        # Using Adam
        self.optimizer = torch.optim.Adam(
            model_params,
            lr=self.train_args.learning_rate,
            weight_decay=self.train_args.weight_decay,
            amsgrad=True)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.train_args.lr_step_epoch,
            gamma=self.train_args.lr_gamma)
        self.model_params = model_params

        self.steps_per_epoch = dataset_len // self.train_args.batch_size

        self.cur_epoch = 1
        self.epoch_step = 1
        self.total_step = 1

        self.prev_time = time.time()
        self.train_time = []

    def generate_state_dict(self):
        # TODO handle LR scheduler
        state_dict = {}
        state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        state_dict['scheduler_state_dict'] = self.scheduler.state_dict()
        state_dict['cur_epoch'] = self.cur_epoch
        state_dict['epoch_step'] = self.epoch_step
        state_dict['total_step'] = self.total_step

        return state_dict

    def load_state_dict(self, state_dict, args):
        # For restarting at new learning rate
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        print(f'Setting new learning rate to {args.train_args.learning_rate}')
        for g in self.optimizer.param_groups:
            g['lr'] = args.train_args.learning_rate
        
        # For re-setting learning rate scheduler
        if 'gamma' in state_dict['scheduler_state_dict']:
            state_dict['scheduler_state_dict']['gamma'] = \
                args.train_args.lr_gamma
        if 'step_size' in state_dict['scheduler_state_dict']:
            state_dict['scheduler_state_dict']['step_size'] = \
                args.train_args.lr_step_epoch                
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])

        # Things intrinsic to our own optimizer
        self.cur_epoch = state_dict['cur_epoch']
        self.epoch_step = state_dict['epoch_step']
        self.total_step = state_dict['total_step']


    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def record_train(self):
        cur_time = time.time()
        self.train_time.append(cur_time - self.prev_time)
        self.prev_time = cur_time

    def print_train_status(self):
        if self.total_step % self.train_args.disp_steps == 0:
            avg_time = round(np.sum(self.train_time) / len(self.train_time), 2)
            print(f'Epoch {self.cur_epoch}.\t'
                  f'Epoch Step {self.epoch_step}/{self.steps_per_epoch}.\t'
                  f'Total Step {self.total_step}\t'
                  f'Avg Time {avg_time}')
            self.train_time = []

    def end_iter(self):
        self.record_train()
        self.print_train_status()
        self.epoch_step += 1
        self.total_step += 1

    def end_epoch(self):
        self.scheduler.step()
        self.cur_epoch += 1
        self.epoch_step = 1

    def completed(self):
        return self.cur_epoch >= self.train_args.max_epochs
