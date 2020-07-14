import sys
sys.path.append('.')

import os
import random

from pathlib import Path
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from constants import *
from utils import *


def select_segment(data, length=5000, avoid=[(-1, -1)]):
    """Select a random segment from data.

    Avoid a certain section (for validation or testing).
    Avoids should also have length of input length
    """

    total_length = len(data)
    half = length // 2
    while True:
        start = np.random.randint(low=0, high=total_length - length)
        end = start + length
        valid = True
        for a in avoid:
            if start >= a[0] and start < a[1] - half:
                valid = False
            if end >= a[0] and end < a[1] and end >= a[1] - half:
                valid = False

        if valid:
            break

    segment = data[start: end]
    return segment


def view_segment(segment, ax):

    x = segment['POSITION_X']
    y = segment['POSITION_Y']
    z = segment['POSITION_Z']

    ax.plot(x, y, z)


def view_path(data):

    plt.ion()

    x = data['POSITION_X']
    y = data['POSITION_Y']
    z = data['POSITION_Z']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.set_zlim(0, 20)

    plt.show()
    return ax


if __name__ == '__main__':
    files = os.listdir(WITH_COLLISIONS_PATH)
    user = pd.read_csv(Path(WITH_COLLISIONS_PATH) / files[0])

    # select_segment(user)

    ax = view_path(user)

    while True:
        useless = input('Enter whatever')
        segment = select_segment(user)
        view_segment(segment, ax)
