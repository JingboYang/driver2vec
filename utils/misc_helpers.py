import time
import numpy as np

from constants import *


def select_columns(spec=None):

    if spec is not None:
        selected = COLUMN_SELECTION_OPTIONS[spec]
    else:
        np.random.seed(int(time.time()))
        all_cols = np.array(USEFULL_COLS)
        selected = np.random.choice(all_cols,
                                    size=NUM_SELECTED_COLS, replace=False)

    return list(selected)


def format_list(items, sep=',', line_len=80, sort=True,
                val_func=lambda x: x, fmt_func=lambda x: x):
    if sort:
        items = sorted(items)
    items = [fmt_func(str(val_func(i))) for i in items]

    output = ''
    cur_line_len = 0
    for it in items:
        if len(it) + cur_line_len > line_len:
            cur_line_len = len(it)
            output += '\n' + it
        else:
            if cur_line_len == 0:
                output += it
                cur_line_len += len(it)
            else:
                output += ', ' + it
                cur_line_len += len(it) + 2
    output += '\n'
    return output

