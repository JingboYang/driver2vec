"""
A lot of the following lines are directly ported from
AI for Healthcare Bootcamp 2019 Winter - 2019 Spring for EEG

Credits also to authors of CheXpert
"""
import sys
sys.path.append('.')

import argparse
import copy
import datetime
import getpass
import json
import os
import pprint as pp
import socket

import pathlib
from pathlib import Path
from pytz import timezone

from constants import *
from logger import Logger
from utils import str_to_bool, GCOpen, GCStorage


class ArgParser(object):
    """Base argument parser for args shared between test and train modes."""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Model for Driver2Vec')

        # Miscellaneous Arguments        
        self.parser.add_argument('--exp_name',
                                 dest='misc_args.exp_name',
                                 type=str, default='default_exp',
                                 help='Experiment name')

        self.parser.add_argument('--device',
                                 dest='misc_args.device',
                                 type=str, default='cuda',
                                 help='Device to use (cpu, cuda)')

        # Data Arguments
        self.parser.add_argument('--dataset',
                                 dest='data_args.dataset',
                                 type=str, default='triplet',
                                 help='Dataset to use')
        
        self.parser.add_argument('--dataset_version',
                                 dest='data_args.dataset_version',
                                 type=str, default='divided_splitted_normalized',
                                 choices=tuple(DATASET_EVAL_METRICS.keys()),
                                 help='Version of dataset on GCP bucket')
        
        self.parser.add_argument('--force_data_update',
                                 dest='data_args.force_data_update',
                                 type=str_to_bool, default=False,
                                 help='Dataset to use')
        
        self.parser.add_argument('--data_spec',
                                 dest='data_args.data_spec',
                                 type=str, default='no_fog_rainsnow_headlight',
                                 choices=list(COLUMN_SELECTION_OPTIONS.keys()),
                                 help='Column selection spec')
        
        self.parser.add_argument('--num_workers',
                                 dest='data_args.num_workers',
                                 type=int, default=4,
                                 help='Number of data fetching workers')
        
        self.parser.add_argument('--segment_gap',
                                 dest='data_args.segment_gap',
                                 type=int, default=50,
                                 help='Selection Ratio')
        
        # Training Arguments
        self.parser.add_argument('--fast_debug',
                                 dest='train_args.fast_debug',
                                 type=str_to_bool, default=False,
                                 help='Shorten DL cycle to cover code faster')

        self.parser.add_argument('--do_train',
                                 dest='train_args.do_train',
                                 type=str_to_bool, default=True,
                                 help='Do training')

        self.parser.add_argument('--do_eval',
                                 dest='train_args.do_eval',
                                 type=str_to_bool, default=True,
                                 help='Do validation')

        self.parser.add_argument('--do_test',
                                 dest='train_args.do_test',
                                 type=str_to_bool, default=True,
                                 help='Do testing')

        self.parser.add_argument('--saved_model_name',
                                 dest='misc_args.saved_model_name',
                                 type=str, default='',
                                 help='Date and name of model, for example'
                                      '2019-04-22/default-exp-12')

        self.parser.add_argument('--batch_size',
                                 dest='train_args.batch_size',
                                 type=int, default=384,
                                 help='Batch siz. 7680 for TCN on T4. '
                                      '4096 for TCN on K80.')

        self.parser.add_argument('--learning_rate',
                                 dest='train_args.learning_rate',
                                 type=float, default=0.0001,
                                 help='Learning rate')

        self.parser.add_argument('--weight_decay',
                                 dest='train_args.weight_decay',
                                 type=float, default=0.00001,
                                 help='L2 penalty multiplier')

        self.parser.add_argument('--lr_step_epoch',
                                 dest='train_args.lr_step_epoch',
                                 type=float, default=4,
                                 help='Number of epochs between stepping '
                                      'down learning rate')

        self.parser.add_argument('--lr_gamma',
                                 dest='train_args.lr_gamma',
                                 type=float, default=0.9,
                                 help='Multiplicative factor of LR decay')

        self.parser.add_argument('--loss_fn_name',
                                 dest='train_args.loss_fn_name',
                                 type=str, default='cross_entropy',
                                 choices=('cross_entropy', 'triplet', 'both'),
                                 help='Name of loss function')

        self.parser.add_argument('--disp_steps',
                                 dest='train_args.disp_steps',
                                 type=int, default=10,
                                 help='Number of steps between displaying'
                                      'training losses')

        self.parser.add_argument('--heavy_log_steps',
                                 dest='train_args.heavy_log_steps',
                                 type=int, default=100,
                                 help='Number of steps between heavy logs, '
                                      'like t-SNE and confusion matrix')

        self.parser.add_argument('--eval_steps',
                                 dest='train_args.eval_steps',
                                 type=int, default=400,
                                 help='Number of steps between evaluations')

        self.parser.add_argument('--save_steps',
                                 dest='train_args.save_steps',
                                 type=int, default=800,
                                 help='Number of steps between saves')

        self.parser.add_argument('--max_epochs',
                                 dest='train_args.max_epochs',
                                 type=int, default=100,
                                 help='Max number of epochs')

        self.parser.add_argument('--triplet_margin',
                                 dest='train_args.triplet_margin',
                                 type=float, default=1.0,
                                 help='Margin for Triplet Loss')

        self.parser.add_argument('--triplet_weight',
                                 dest='train_args.triplet_weight',
                                 type=float, default=0.5,
                                 help='Weight for Triplet Loss')

        self.parser.add_argument('--clipping_value',
                                 dest='train_args.clipping_value',
                                 type=float, default=1.0,
                                 help='Gradient clipping value')

        # Model Arguments
        self.parser.add_argument('--model_name',
                                 dest='model_args.model_name',
                                 type=str, default='TCN',
                                 help='Name of model to use')

        self.parser.add_argument('--do_triplet',
                                 dest='model_args.do_triplet',
                                 type=str_to_bool, default=True,
                                 help='Whether to use triplet')

        self.parser.add_argument('--wavelet',
                                 dest='model_args.wavelet',
                                 type=str_to_bool, default=False,
                                 help='Whether to include the '
                                      'wavelet variables')

        self.parser.add_argument('--wavelet_output_size',
                                 dest='model_args.wavelet_output_size',
                                 type=int, default=15,
                                 help='output size of linear layer '
                                      'for the wavelet step')

        self.parser.add_argument('--input_length',
                                 dest='model_args.input_length',
                                 type=int, default=1000,
                                 help='Length of input')

        self.parser.add_argument('--input_channels',
                                 dest='model_args.input_channels',
                                 type=int, default=23,
                                 help='Number of channels in input')

        self.parser.add_argument('--output_size',
                                 dest='model_args.output_size',
                                 type=int, default=NUM_DRIVERS,
                                 help='Number of outputs')
                                      
        self.parser.add_argument('--rnn_model',
                                 dest='model_args.rnn_model',
                                 choices=['LSTM', 'GRU'],
                                 type=str,
                                 default="GRU",
                                 help='LSTM or GRU')

        self.parser.add_argument('--rnn_use_last',
                                 dest='model_args.rnn_use_last',
                                 type=str_to_bool, default=True,
                                 help='True or False')

        self.parser.add_argument('--dropout',
                                 dest='model_args.dropout',
                                 type=float,
                                 default=0.1,
                                 help='Dropout probability')

        self.parser.add_argument('--kernel_size',
                                 dest='model_args.kernel_size',
                                 type=int, default=7,
                                 help='Kernel Size')

        self.parser.add_argument('--channel_lst',
                                 dest='model_args.channel_lst',
                                 type=str,
                                 default='25,25,25,25,25,25,25,25',
                                 help='String to store hidden size '
                                 'in each layer ')

        self.parser.add_argument('--rnn_hidden_size',
                                 dest='model_args.rnn_hidden_size',
                                 type=int, default=64,
                                 help='hidden size for RNN model')

        self.parser.add_argument('--rnn_num_layers',
                                 dest='model_args.rnn_num_layers',
                                 type=int, default=2,
                                 help='number of layers for RNN model')

    @staticmethod
    def namespace_to_dict(args):
        """Turn a nested Namespace object into a nested dictionary."""
        args_dict = vars(copy.deepcopy(args))

        for arg in args_dict:
            obj = args_dict[arg]
            if isinstance(obj, argparse.Namespace):
                item = ArgParser.namespace_to_dict(obj)
                args_dict[arg] = item
            else:
                if isinstance(obj, pathlib.PosixPath):
                    args_dict[arg] = str(obj)

        return args_dict

    # Only one level of nesting is supported.
    def fix_nested_namespaces(self, args):
        """Convert a Namespace object to a nested Namespace."""
        group_name_keys = []

        for key in args.__dict__:
            if '.' in key:
                group, name = key.split('.')
                group_name_keys.append((group, name, key))

        for group, name, key in group_name_keys:
            if group not in args:
                args.__dict__[group] = argparse.Namespace()

            args.__dict__[group].__dict__[name] = args.__dict__[key]
            del args.__dict__[key]

    def get_experiment_number(self, experiments_dir, experiment_name):
        """Parse directory to count the previous copies of an experiment."""
        dir_structure = GCStorage.MONO.list_files(experiments_dir)
        dirnames = [exp_dir.split('/')[-1] for exp_dir in dir_structure[1]]

        ret = 1
        for d in dirnames:
            if d[:d.rfind('_')] == experiment_name:
                ret = max(ret, int(d[d.rfind('_') + 1:]) + 1)
        return ret

    def parse_args(self):
        """Parse command-line arguments and create directories."""
        args = self.parser.parse_args()

        # Make args a nested Namespace.
        self.fix_nested_namespaces(args)
        
        us_timezone = timezone('US/Pacific')
        date = datetime.datetime.now(us_timezone).strftime("%Y-%m-%d")
        save_dir = Path(EXP_STORAGE) / date

        args.exp_name = getpass.getuser() + '_' + socket.gethostname() + \
                        '_KGAT_' + args.misc_args.exp_name + '_' + \
                        args.data_args.dataset
        exp_num = self.get_experiment_number(save_dir, args.misc_args.exp_name)

        args.exp_name = args.misc_args.exp_name + '_' + str(exp_num)
        save_dir = save_dir / args.misc_args.exp_name
        log_file = save_dir / 'run_log.txt'

        arg_dict = self.namespace_to_dict(args)
        arg_dict_text = pp.pformat(arg_dict, indent=4)

        arg_text = ' '.join(sys.argv)

        args.misc_args.logger = Logger(log_file, save_dir)

        args.misc_args.logger.log_text(\
                            {'setup:command_line': arg_text,
                             'setup:parsed_arguments': arg_dict_text},
                             0)
        
        # Temporarily disable model saving/loading
        #args.misc_args.model_save_dir = \
        #    args.misc_args.heavy_save_dir / 'checkpoints'
        #args.misc_args.model_save_dir.mkdir(exist_ok=True)

        return args

if __name__ == '__main__':
    parser = ArgParser()
    parser.parse_args()
