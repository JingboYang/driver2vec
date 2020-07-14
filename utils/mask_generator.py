import itertools
import numpy as np
from scipy.special import comb

from constants import MAX_P_WAYS, NUM_DRIVERS

    
SAMPLE_SIZE = 1000

def findsubsets(S,m, sample_size):
    """
    Generate sll combination of size m that is a subset of set S

    args:
        S: iterable
        m: size of the subsets
        sample_size: only return sample_size subsets

    returns:
        a set of tuples
    """
    all_combos = np.array(list(itertools.combinations(S, m)))
    if len(all_combos) > sample_size:
        indices = np.random.choice(list(np.arange(len(all_combos))),
                                   size=sample_size,
                                   replace=False)
        all_combos = list(all_combos[indices])

    return list(all_combos)


def generate_mask_helper(n, driver):
    """
    p: number of drivers including the correct driver >>>p way accuracy
    if say current driver is i, then among all the other drivers, create 
    a mask of  all enumerations of the (n-1) choose (p-1) combinations.
    
    e.g. driver 1(0 indexed) and p=2-->
    [[0,1,1,0,....,0],[0,1,0,1,....,0], ..., [0,1,0,0,....,1]] len=(n-1) choose (p-1)
    """
    np.random.seed(341)

    sample_size = SAMPLE_SIZE
    masks = dict()
    drivers = set(np.arange(n))
    for p in range(2, MAX_P_WAYS + 1, 1):
        drivers_withoutdriver = drivers - set([driver])

        combinations = findsubsets(drivers_withoutdriver, p - 1, sample_size)
        mask_driver = np.zeros((len(combinations), n))
        mask_driver[:, driver] = 1

        row = 0
        for i in combinations:
            # print(row, i)
            mask_driver[row, i] = 1
            row += 1
        mask_driver = mask_driver.astype(int)
        masks[p] = mask_driver
    return masks


class MaskGenerator:
    
    MONO = None

    @staticmethod
    def get_MaskGenerator(*args, **kwargs):
        if MaskGenerator.MONO is not None:
            return MaskGenerator.MONO
        else:
            MaskGenerator.MONO = MaskGenerator(*args, **kwargs)
            return MaskGenerator.MONO

    def __init__(self):
        
        self.mask_drivers = dict()

        # TODO Make this less "exposed" here
        for driver in range(NUM_DRIVERS):
            # print("==> Generating mask for driver {}".format(driver))
            self.mask_drivers[driver] = generate_mask_helper(NUM_DRIVERS, driver)
        print('Generated masks')
    
    def get_mask(self):
        return self.mask_drivers

