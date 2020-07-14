"""
A lot of the following lines are directly ported from
AI for Healthcare Bootcamp 2019 Winter - 2019 Spring for EEG

Credits also to authors of CheXpert
"""

import torch
import numpy as np
from collections import defaultdict

from tqdm import tqdm
import lightgbm as lgb

from constants import NUM_DRIVERS
from logger import *


def recursive_append(target_dict, source_dict):    
    for e in source_dict:
        if type(source_dict[e]) == dict:
            if e not in target_dict:
                target_dict[e] = defaultdict(list)
            target_dict[e] = recursive_append(target_dict[e], source_dict[e])
        elif source_dict[e] is not None:
            if type(source_dict[e]) == list:
                target_dict[e].append(source_dict[e])
            else:
                target_dict[e].append(source_dict[e].cpu())
    
    return target_dict

def recursive_concat(source_dict):
    for e in source_dict:
        if type(source_dict[e]) == dict or type(source_dict[e]) == defaultdict:
            source_dict[e] = recursive_concat(source_dict[e])
        elif source_dict[e] is not None:
            source_dict[e] = np.concatenate(source_dict[e])
    
    return source_dict

class Predictor(object):
    """Predictor class for a single model."""

    def __init__(self, model, device, fast_debug):
        self.model = model
        self.device = device
        self.fast_debug = fast_debug
        self.logger = Logger.get_unique_logger()

        self.train_results = None

        # TODO Move this to constants or to command line
        # Maybe easier to have a spec file for LightGBM
        # https://stackoverflow.com/questions/47370240/multiclass-classification-with-lightgbm
        # https://lightgbm.readthedocs.io/en/latest/Python-Intro.html#training
        self.lgb_param = {'num_leaves': 31,
                          'num_trees': 100,
                          'boosting_type': 'gbdt',
                          'objective': 'multiclass',
                          'num_class': NUM_DRIVERS,
                          'max_depth': 12,
                          'verbosity': 0,
                          'task': 'train',
                          'metric': 'multi_logloss',
                          "learning_rate" : 1e-2,
                          "bagging_fraction" : 0.9,  # subsample
                          "bagging_freq" : 5,        # subsample_freq
                          "bagging_seed" : 341,
                          "feature_fraction" : 0.8,  # colsample_bytree
                          "feature_fraction_seed":341,}
        self.lgb_num_rounds = 15

    def _predict(self, loader, ratio=1, need_triplet_emb=True):

        self.model.eval()

        outputs = []
        ground_truth = []
        other_info = defaultdict(list)
        debug_counter = 0
        with tqdm(total=len(loader.dataset)) as progress_bar:
            for orig_features, pos_features, neg_features, targets, data_info \
                in loader:

                if np.random.rand() > ratio:
                    progress_bar.update(targets.size(0))
                    continue

                with torch.no_grad():
                    predictions, info = self.model(orig_features,
                                                  pos_features,
                                                  neg_features,
                                                  need_triplet_emb)

                outputs.append(predictions.cpu())
                ground_truth.append(targets.cpu())
                # embeddings.append(emb.cpu())
                other_info = recursive_append(other_info, info)
                if 'data_info' not in other_info:
                    other_info['data_info'] = recursive_append(
                        defaultdict(list),
                        data_info)
                else:
                    other_info['data_info'] = recursive_append(
                        other_info['data_info'],
                        data_info)
                progress_bar.update(targets.size(0))

                debug_counter += 1
                if self.fast_debug and debug_counter >= 4:
                    break

        outputs = np.concatenate(outputs)
        ground_truth = np.concatenate(ground_truth)
        other_info = recursive_concat(other_info)

        self.model.train()

        return outputs, ground_truth, other_info

    def start_prediction(self, train_loader,
                         save_train_emb=True):
        # TODO Let's say, we want no more than 10K input for LightGBM
        data_count = len(train_loader.dataset)
        NUM_ALLOWED = 20000.0
        if data_count > NUM_ALLOWED:
            ratio = NUM_ALLOWED / data_count
            self.logger.log(f'Using ratio of {ratio} for '\
                            f'total count of {data_count}')
        else:
            ratio = 1.0
            self.logger.log(f'Full set is used')
        train_out, train_gt, train_emb = self._predict(train_loader, ratio,
                                                       False)
        self.train_results = train_out, train_gt, train_emb

        if save_train_emb:
            self.logger.log_data(train_emb, f'train_embeddings_latest')
            self.logger.log_data(train_gt, f'train_ground_truth_latest')
    
    def named_predict(self, loader_name, other_loader, pred_name):
        func_name = '_' + pred_name
        return getattr(self, func_name)(other_loader, loader_name)

    def _simple_predict(self, other_loader, loader_name, 
                              save_simple_predict=True):
        other_out, other_gt, other_info = self._predict(other_loader)

        if save_simple_predict:
            self.logger.log_data(other_info,
                                 f'eval_{loader_name}_embeddings_latest')
            self.logger.log_data(other_gt,
                                 f'eval_{loader_name}_ground_truth_latest')

        return {'predictions': other_out,
                'ground_truth': other_gt,
                'other_info': other_info}

    def _lgbm_predict(self, other_loader, loader_name,
                            save_simple_predict=True):
        other_out, other_gt, other_info = self._predict(other_loader)
        train_out, train_gt, train_emb = self.train_results

        if save_simple_predict:
            self.logger.log_data(other_info,
                                 f'eval_{loader_name}_embeddings_latest')
            self.logger.log_data(other_gt,
                                 f'eval_{loader_name}_ground_truth_latest')

        train_data = lgb.Dataset(train_emb['orig'], label=train_gt)
        bst = lgb.train(self.lgb_param, train_data, self.lgb_num_rounds)
        other_bst_out = bst.predict(other_info['orig'])

        return {'predictions': other_bst_out,
                'ground_truth': other_gt,
                'other_info': other_info}
    
    def end_prediction(self):
        self.train_results = None
