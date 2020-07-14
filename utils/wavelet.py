import pywt
import numpy as np

def gen_wavelet(data):
    ncol = data.shape[1]
    nrow = data.shape[0]
    for i in range(ncol):
        cur_col = data[:,i].copy()
        (cA, cD) = pywt.dwt(cur_col, 'haar')
        new_col = np.reshape(np.concatenate((cA,cD), 0),(nrow,1))
        data = np.hstack((data,new_col))
    return data