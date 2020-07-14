import os
import sys
sys.path.append('.')

# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
# Solve the issue of RuntimeError: received 0 items of ancdata when loading
# very large dataset and GPU cannot catch up with worker speed
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

from collections import defaultdict
import pickle
import random

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from constants import *
from data import *
from utils import *
from data import *
from logger import *


# Set random seed so that the data splitting is deterministic
np.random.seed(0)
random.seed(0)

class TripletDataset(Dataset):

    def __init__(self, args, split, same_area, ss_class):
        '''
        TODO: Handle same_area later
        '''
        super(TripletDataset, self).__init__()

        self.logger = Logger.get_unique_logger()

        self.data_args = args.data_args
        self.model_args = args.model_args
        self.wavelet = self.model_args.wavelet

        self.split = split
        self.keep_same_area = same_area
        self.segment_selector = ss_class(args, split)        

    def append(self, *args, **kwargs):
        self.segment_selector.append(*args, **kwargs)

    def summarize(self):
        self.logger.log('TODO! Generate some summary for triplet dataset.')
        pass

    def __len__(self):
        return len(self.segment_selector)

    def __getitem__(self, idx):
        
        orig_seg, orig_meta = \
                self.segment_selector.select_segment('index', index=idx)
        orig_driver = orig_meta['driver']
        orig_area = orig_meta['area']
        pos_seg, pos_meta = \
            self.segment_selector.select_segment('driver_area',
                                                 inc_driver=orig_driver,
                                                 exc_driver=None,
                                                 area=None)
        neg_seg, neg_meta = \
            self.segment_selector.select_segment('driver_area',
                                                 inc_driver=None,
                                                 exc_driver=orig_area,
                                                 area=None)
        neg_driver = neg_meta['driver']
        neg_area = neg_meta['area']

        if self.wavelet:
            out1 = gen_wavelet(np.array(orig_seg, dtype=np.float32))
            out2 = gen_wavelet(np.array(pos_seg, dtype=np.float32))
            out3 = gen_wavelet(np.array(neg_seg, dtype=np.float32))
        else:
            out1 = np.array(orig_seg, dtype=np.float32)
            out2 = np.array(pos_seg, dtype=np.float32)
            out3 = np.array(neg_seg, dtype=np.float32)

        # self.logger.log(f'Orig Length: {len(out1)}')
        # self.logger.log(f'Pos Length: {len(out2)}')
        # self.logger.log(f'Neg Length: {len(out3)}')

        # TODO: Handle collision later
        this_collision = None

        description = (f'Driver: {orig_driver}, '
                       f'Area: {orig_area}, '
                       f'index: {idx}, '
                       f'collision: {this_collision} ')
        return (out1,
                out2,
                out3,
                np.array(orig_driver),
                {'mask': {},
                 'description': description,
                 'other_driver_gt': neg_area}
                )

class FullDataDatasetHandler():

    def __init__(self, args, whole_data, useful_cols, dataset_class):
        self.logger = Logger.get_unique_logger()

        self.args = args
        self.whole_data = whole_data
        self.useful_cols = useful_cols

        train = dataset_class(
                                args=args, split='train',
                                same_area=False, ss_class=DriverRandomSS
                             )

        eval_simple = dataset_class(
                                args=args, split='eval_simple',
                                same_area=False, ss_class=DriverRandomSS
                                )
        eval_lgbm = dataset_class(
                                args=args, split='eval_lgbm',
                                same_area=False, ss_class=DriverRandomSS
                                )
        
        test_simple = dataset_class(
                                args=args, split='test_simple',
                                same_area=False, ss_class=DriverRandomSS
                                )
        test_lgbm = dataset_class(
                                args=args, split='test_lgbm',
                                same_area=False, ss_class=DriverRandomSS
                                )

        # MUST match name of DATA_EVAL_METRICS
        self.datasets = {'train': {'train': train},
                         'eval': {'eval_simple': eval_simple,
                                  'eval_lgbm': eval_lgbm},
                         'test': {'test_simple': test_simple,
                                  'test_lgbm': test_lgbm}}
                         # TODO Fixed area for test
        self.generate()

        for d in self.datasets:
            if type(self.datasets[d]) != dict:
                self.datasets[d].summarize()
            else:
                for dd in self.datasets[d]:
                    self.datasets[d][dd].summarize()

    def generate(self):

        for user in self.whole_data:
            for split in self.whole_data[user]:
                for area in self.whole_data[user][split]:
                    for data in self.whole_data[user][split][area]:

                        driver_index = LABEL_TO_INDEX[str(user)]
                        area_id = area

                        if split == 'train':
                            self.datasets['train']['train'].append(
                                driver_index, area_id,
                                data['data'][self.useful_cols])
                        elif split == 'eval':
                            self.datasets['eval']['eval_simple'].append(
                                driver_index, area_id,
                                data['data'][self.useful_cols])
                            self.datasets['eval']['eval_lgbm'].append(
                                driver_index, area_id,
                                data['data'][self.useful_cols])
                        elif split == 'test':
                            self.datasets['test']['test_simple'].append(
                                driver_index, area_id,
                                data['data'][self.useful_cols])
                            self.datasets['test']['test_lgbm'].append(
                                driver_index, area_id,
                                data['data'][self.useful_cols])

        step_per_epoch = (len(self.datasets['train']) /
                          self.args.train_args.batch_size)
                          
        self.logger.log(
            f'There is {round(step_per_epoch, 2)} steps per epoch.')
        self.logger.log(
            f'\t For {len(self.datasets["train"])} samples with batch size '
            f'{self.args.train_args.batch_size}')
                 

class DatasetManager():
    """Manage dataset creation and dataloader creation.

    We need a manager to generate train/val/test segments.
    This would be terribly messy without a class-level manager.
    """

    def __init__(self, args, useful_cols):
        self.args = args
        self.useful_cols = useful_cols

        self.ds_handler = None
        self.dataloaders = None

        self.download_dataset(self.args.data_args.force_data_update)

    def download_dataset(self, force_update):
        gc_dataset_folder = Path(DATA_STORAGE) / \
                                            self.args.data_args.dataset_version
        local_dataset_folder = Path(TEMP_FOLDER) / Path(DATA_STORAGE) / \
                                            self.args.data_args.dataset_version

        os.makedirs(local_dataset_folder, exist_ok=True)

        user_data = dict()
        for u in LABEL_TO_INDEX:
            user_fname = f'user_{u}.pickle'
            if force_update or not \
                os.path.exists(local_dataset_folder / user_fname):
                
                GCStorage.MONO.download(local_dataset_folder / user_fname,
                                        gc_dataset_folder / user_fname)

                assert os.path.exists(local_dataset_folder / user_fname)
            
            # import pdb; pdb.set_trace()
            try:
                user_data[int(u)] = \
                    pickle.load(open(local_dataset_folder / user_fname,
                                        'rb'))
            except:
                self.args.misc_args.logger.log(
                                            f'Unable to process {user_fname}')
        
        self.whole_data = user_data

    def get_datasets(self):        
        if self.args.data_args.dataset == 'triplet':
            self.ds_handler = \
                FullDataDatasetHandler(self.args, self.whole_data,
                                       self.useful_cols, TripletDataset)
            return self.ds_handler
        else:
            raise NotImplementedError(f'Dataset {self.args.train_args.dataset}'
                                      ' has not been implemented')

    def get_dataloaders(self):
        self.get_datasets()
        datasets = self.ds_handler.datasets
        self.dataloaders = defaultdict(dict)

        for d in datasets:
            for dd in datasets[d]:
                print(f'Dataset {d}_{dd} has length {len(datasets[d][dd])}')
                self.dataloaders[d][dd] = DataLoader(
                    datasets[d][dd],
                    batch_size=self.args.train_args.batch_size,
                    shuffle=(d == 'train'),
                    num_workers=self.args.data_args.num_workers)

        return self.dataloaders
