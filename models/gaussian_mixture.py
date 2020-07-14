import sys
sys.path.append('.')

from sklearn import mixture

from data import *
from constants import *
from utils import *


def main():

    clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
    all_data = load_data()

    segments = []
    for cl_nc in all_data:
        for d in all_data[cl_nc]:
            data = all_data[cl_nc][d]

            counter = 1000
            while counter > 0:
                d = select_segment(data, length=3000, avoid=[])
                segments.append(d['ACCELERATION'])
                counter -= 1

    clf.fit(segments)


if __name__ == '__main__':
    main()
