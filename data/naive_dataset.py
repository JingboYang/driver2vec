import sys
sys.path.append('.')

# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
# Solve the issue of RuntimeError: received 0 items of ancdata when loading
# very large dataset and GPU cannot catch up with worker speed
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import random
import multiprocessing
import pickle

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from constants import *
from utils import *
from data import *
from logger import *


# Set random seed so that the data splitting is deterministic
np.random.seed(0)
random.seed(0)

# JBY: Cheap way of not importing utils from old branch

def helper(fname):
    # data = pickle.load(fname.open('rb'))
    data = pd.read_csv(fname.open())
    return data, LABEL_TO_INDEX[str(fname).split('_')[-6]]

def load_data(data_dir):

    filenames = []
    for item in data_dir.iterdir():
        # if '.pickle' in str(item):
        fname = data_dir / item
        filenames.append(fname)
    
    pool = multiprocessing.Pool(8)
    loaded = pool.map(helper, filenames)

    print(f'A total of {len(loaded)} items loaded.')

    return loaded


def weighted_random(values, labels):
    total = np.sum(values)
    selection = np.random.randint(total)

    for i, v in enumerate(values):
        if selection < v:
            return (labels[i], selection)
        else:
            selection -= v
##################################################################

class TripletDataset(Dataset):

    def __init__(self, tag, slice_length, wavelet,
                 selection_ratio, same_area=False):
        super(TripletDataset, self).__init__()

        # TODO Add this to arguments. Remember sample is at 100Hz.
        self.selection_ratio = selection_ratio

        self.length = 0
        self.slice_length = slice_length
        self.data = []
        self.keep_same_area = same_area
        self.driver_data_collection = {i: {'length': 0,
                                           'data': [],
                                           'area_id': []}
                                       for i in range(NUM_DRIVERS)}
        self.wavelet = wavelet
        self.tag = tag
        self.logger = Logger.get_unique_logger()

    def append(self, val, driver, areaID=None, collision=None):
        if len(val) > self.slice_length:
            chunk_length = len(val) - self.slice_length
            driver_length = self.driver_data_collection[driver]['length']
            item = (self.length, driver_length, chunk_length, val,
                    driver, areaID, collision)
            self.data.append(item)
            self.driver_data_collection[driver]['data'].append(item)
            self.driver_data_collection[driver]['area_id'].append(areaID)
            self.driver_data_collection[driver]['length'] += chunk_length
            self.length += chunk_length
        else:
            print(f'WARNING! Driver {driver} does not have enough '
                  f'data for {self.tag} at Area {areaID}')

    def summarize(self):
        print('TODO! Generate some summary for triplet dataset.')
        pass

    def check_collision(self, collision_chunk):
        if np.sum(collision_chunk) > (len(collision_chunk) // 2):
            return True
        return False

    def pick_from_driver(self, driver, area_id=None):
        """Pick a random slice from a driver."""
        driver_data = self.driver_data_collection[driver]['data']
        driver_length = self.driver_data_collection[driver]['length']

        # Keep on randomly picking stuff until area id is good
        # TODO Make this process less dumb
        attempt_counter = 0
        while True:
            i = np.random.randint(low=0, high=driver_length)

            for start, driver_start, chunk_length,\
                data, driver, aid, collision\
                in driver_data:
                if i >= driver_start and i < driver_start + chunk_length:
                    # Check if the selection satisfies requirement
                    if self.keep_same_area:
                        if aid != area_id:
                            attempt_counter += 1
                            if attempt_counter > 200:
                                raise LookupError(
                                    f'Could not find area {aid} for driver '
                                    f'{driver} after {attempt_counter} '
                                    f'attempts')
                            break
                        else:
                            pass
                            # print(f'Found area {aid} for driver {driver} '
                            # 'after {attempt_counter} attempts')
                    index = i - driver_start
                    sample = data[index:index + self.slice_length]
                    collision_data = collision[index:index + self.slice_length]
                    has_collision = self.check_collision(collision_data)
                    return sample, aid, has_collision

    def select_random_driver(self, avoid_driver):

        weights = []
        labels = []
        for d in self.driver_data_collection:
            if d != avoid_driver:
                weights.append(self.driver_data_collection[d]['length'])
                labels.append(d)

        selected_driver, selected_index = weighted_random(weights, labels)
        return selected_driver

    def __len__(self):
        return int(self.length / self.selection_ratio)

    def __getitem__(self, i):
        
        # In case we fail to get a set for the same area
        done = False
        while not done:
            i = (int(i * self.selection_ratio) +
                np.random.randint(0, int(self.selection_ratio / 4)))

            for start, driver_start, chunk_length, \
                data, driver, aid, collision\
                in self.data:
                if i >= start and i < start + chunk_length:
                    index = i - start
                    sample = data[index:index + self.slice_length]
                    this_collision = self.check_collision(
                        collision[index:index + self.slice_length])

                    try:
                        if not self.keep_same_area:
                            same_driver, sd_aid, sd_c = self.pick_from_driver(
                                driver, area_id=aid)
                        else:
                            # If for same area, we do not need from same driver
                            same_driver = sample
                            sd_aid = aid
                        od = self.select_random_driver(driver)
                        other_driver, od_aid, od_c = self.pick_from_driver(
                            od, area_id=aid)
                        
                        assert driver != od

                        if self.keep_same_area:
                            assert sd_aid == aid
                            assert od_aid == aid
                        done = True
                    except LookupError:
                        self.logger.log(f'Failed to find area {aid} data '
                                        f'for driver {driver}')
                    break

            i = np.random.randint(low=0, high=len(self))

        if self.wavelet:
            out1 = gen_wavelet(np.array(sample, dtype=np.float32))
            out2 = gen_wavelet(np.array(same_driver, dtype=np.float32))
            out3 = gen_wavelet(np.array(other_driver, dtype=np.float32))
        else:
            out1 = np.array(sample, dtype=np.float32)
            out2 = np.array(same_driver, dtype=np.float32)
            out3 = np.array(other_driver, dtype=np.float32)

        description = (f'Driver: {driver}, '
                       f'Area: {aid}, '
                       f'index: {index}, '
                       f'collision: {this_collision} ')
        return (out1,
                out2,
                out3,
                np.array(driver),
                {'mask': {},
                 'description': description,
                 'other_driver_gt': od}
                )


class FullDataDatasetHandler():

    def __init__(self, args, whole_data, useful_cols, dataset_class):
        self.logger = Logger.get_unique_logger()

        self.whole_data = whole_data
        self.args = args
        self.useful_cols = useful_cols

        train = dataset_class('train',
                              args.model_args.input_length,
                              args.model_args.wavelet,
                              args.data_args.segment_gap)

        eval_simple = dataset_class('eval_simple',
                                    args.model_args.input_length,
                                    args.model_args.wavelet,
                                    args.data_args.segment_gap)
        eval_lgbm = dataset_class('eval_lgbm',
                                  args.model_args.input_length,
                                  args.model_args.wavelet,
                                  args.data_args.segment_gap)
        eval_fixed_area = dataset_class('eval_fixed_area',
                                        args.model_args.input_length,
                                        args.model_args.wavelet,
                                        args.data_args.segment_gap,
                                        same_area=True)

        test_simple = dataset_class('test_simple',
                                    args.model_args.input_length,
                                    args.model_args.wavelet,
                                    args.data_args.segment_gap)
        test_lgbm = dataset_class('test_lgbm',
                                  args.model_args.input_length,
                                  args.model_args.wavelet,
                                  args.data_args.segment_gap)

        # MUST match name of DATA_EVAL_METRICS
        self.datasets = {'train': {'train': train},
                         'eval': {'eval_simple': eval_simple,
                                  'eval_lgbm': eval_lgbm,
                                  'eval_fixed_area': eval_fixed_area},
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

        for data, driver_index in self.whole_data:
            #try:
            #    fname = row['fname']
            #except:
            #    import pdb; pdb.set_trace()
            print(data[self.useful_cols].isnull().values.any())
            # import pdb; pdb.set_trace()

            train_start = 0
            train_end = 8*(len(data[self.useful_cols]) // 10)
            
            eval_start = 8*(len(data[self.useful_cols]) // 10)
            eval_end = 9*(len(data[self.useful_cols]) // 10)
            
            test_start = 9*(len(data[self.useful_cols]) // 10)
            test_end = len(data[self.useful_cols])

            area_id = -1

            self.datasets['train']['train'].append(
                data[self.useful_cols][train_start:train_end],
                driver_index, area_id,
                [0] * len(data[self.useful_cols][train_start:train_end]))
            
            self.datasets['eval']['eval_simple'].append(
                data[self.useful_cols][eval_start:eval_end],
                driver_index, area_id,
                [0] * len(data[self.useful_cols][eval_start:eval_end]))
            
            self.datasets['eval']['eval_lgbm'].append(
                data[self.useful_cols][eval_start:eval_end],
                driver_index, area_id,
                [0] * len(data[self.useful_cols][eval_start:eval_end]))
            
            self.datasets['eval']['eval_fixed_area'].append(
                data[self.useful_cols][eval_start:eval_end],
                driver_index, area_id,
                [0] * len(data[self.useful_cols][eval_start:eval_end]))
            
            self.datasets['test']['test_simple'].append(
                data[self.useful_cols][test_start:test_end],
                driver_index, area_id,
                [0] * len(data[self.useful_cols][test_start:test_end]))

            self.datasets['test']['test_lgbm'].append(
                data[self.useful_cols][test_start:test_end],
                driver_index, area_id,
                [0] * len(data[self.useful_cols][test_start:test_end]))
        
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
        local_dataset_folder = Path(TEMP_FOLDER) / Path(DATA_STORAGE) / \
                                            self.args.data_args.dataset_version
        self.whole_data = load_data(local_dataset_folder)
        self.useful_cols = useful_cols

        self.ds_handler = None
        self.dataloaders = None

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

        # import pdb; pdb.set_trace()

        for d in datasets:
            for dd in datasets[d]:
                print(f'Dataset {d}_{dd} has length {len(datasets[d][dd])}')
                self.dataloaders[d][dd] = DataLoader(
                    datasets[d][dd],
                    batch_size=self.args.train_args.batch_size,
                    shuffle=(d == 'train'),
                    num_workers=self.args.data_args.num_workers)

        return self.dataloaders
