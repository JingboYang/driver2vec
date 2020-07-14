from collections import defaultdict
import random

from constants import *
from utils import *
from logger import *


class BaseSegmentSelector:
    '''Base class for selecting a driving segment from data'''

    def __init__(self, args, split):
        self.logger = Logger.get_unique_logger()
        
        self.data_args = args.data_args
        self.model_args = args.model_args

        self.seg_length = self.model_args.input_length
        self.seg_gap = self.data_args.segment_gap

        # Number of valid segments
        # (used in for loop for driver_train.py and for dataset class)
        self.num_valid_segments = 0
        self.split = split

        self.valid_areas = [1, 8, 9]

    def append(self, driver, areaID, data, **kwargs):
        '''Used when adding a chunk of data that is loaded from pickle'''
        raise NotImplementedError

    def select_segment(self, method, **kwargs):
        '''Return a segment according to requirement (e.g. same driver)'''
        func_name = f'_select_segment_by_{method}'
        return getattr(self, func_name)(**kwargs)

    def __len__(self):
        return self.num_valid_segments
    

class DriverRandomSS(BaseSegmentSelector):
    '''Random selection of data segment.

    Could be more efficient if a proper binary index tree was used
    '''
    def __init__(self, args, split):
        super(DriverRandomSS, self).__init__(args, split)
        # Remember to iterate through valid areas as dictionary keys
        self.index_data_collection = defaultdict(list)
        self.driver_data_collection = defaultdict(lambda :defaultdict(list))

    def append(self, driver, area, data):
        # import pdb; pdb.set_trace()

        if len(data) > self.seg_length:
            chunk_length = len(data) - self.seg_length
            num_segs = chunk_length // self.seg_gap

            item = {'driver': driver, 'area': area,
                    'num_segs': num_segs, 'data': data}
            self.index_data_collection[area].append(item)
            self.driver_data_collection[driver][area].append(item)
            self.num_valid_segments += num_segs
        else:
            # import pdb; pdb.set_trace()
            self.logger.log(f'WARNING! '
                            f'Driver {driver} does not have enough '
                            f'data for {self.split} at Area {area}')
    
    def _select_segment_by_index(self, index):
        '''Select data by iterating index'''
        selected = None
        meta = None
        current = 0

        # import pdb; pdb.set_trace()

        for area in self.index_data_collection:
            for item in self.index_data_collection[area]:
                prev_len = current
                current = prev_len + item['num_segs']
                if index >= prev_len and index < current:
                    inner_start = (index - prev_len) * self.seg_gap
                    inner_end = inner_start + self.seg_length

                    selected = item['data'][inner_start:inner_end]
                    meta = {'driver': item['driver'], 'area': item['area']}
                    break
            if selected is not None:
                break
        return selected, meta

    def _select_segment_by_driver_area(self, inc_driver, exc_driver, area):
        '''Select a segment from NOT a certain driver

        If inc_driver is None then everyone is included
        If exc_driver is None then nobody is included

        If inc is None exc is not None, then exc is actually excluded
        If inc is not None and exc is None, then inc is actually included
        If both None then everyone is included
        If inc is not None and exc is not None, then, same as just inc

        If area is None then any area is okay
        '''
        if area is not None and area not in self.valid_areas:
            raise ValueError(f'Area {area} is not allowed')
        if inc_driver is not None and exc_driver is not None:
            assert inc_driver != exc_driver, \
                    f'Cannot include and exclue {inc_driver} at the same time'

        selected = None
        meta = None
        current = 0

        if area is not None:
            total_length = 0
            for driver in self.driver_data_collection:
                if driver is exc_driver:
                    continue
                if inc_driver is None or inc_driver == driver:
                    for item in self.driver_data_collection[driver][area]:
                        total_length += item['num_segs']
            
            index = random.randint(0, total_length - 1)
            for driver in self.driver_data_collection:
                if driver is exc_driver:
                    continue
                if inc_driver is None or inc_driver == driver:
                    for item in self.driver_data_collection[driver][area]:
                        prev_len = current
                        current = prev_len + item['num_segs']
                        if index >= prev_len and index < current:
                            inner_start = (index - prev_len) * self.seg_gap
                            inner_end = inner_start + self.seg_length

                            selected = item['data'][inner_start:inner_end]
                            meta = {'driver': item['driver'],
                                    'area': item['area']}
                            break
                    if selected is not None:
                        break
        else:
            total_length = 0
            for driver in self.driver_data_collection:
                if driver is exc_driver:
                    continue
                if inc_driver is None or inc_driver == driver:
                    for area in self.driver_data_collection[driver]:
                        for item in self.driver_data_collection[driver][area]:
                            total_length += item['num_segs']
            
            index = random.randint(0, total_length - 1)
            for driver in self.driver_data_collection:
                if driver is exc_driver:
                    continue
                if inc_driver is None or inc_driver == driver:
                    for area in self.driver_data_collection[driver]:
                        for item in self.driver_data_collection[driver][area]:
                            prev_len = current
                            current = prev_len + item['num_segs']
                            if index >= prev_len and index < current:
                                inner_start = (index - prev_len) * self.seg_gap
                                inner_end = inner_start + self.seg_length

                                selected = item['data'][inner_start:inner_end]
                                meta = {'driver': item['driver'],
                                        'area': item['area']}
                                break
                        if selected is not None:
                            break
                if selected is not None:
                        break
        
        return selected, meta