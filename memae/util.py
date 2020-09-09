import os
import h5py
import random
import numpy as np
import torch


def get_dataset(dirname=os.path.join('hdf5'), filename=None, tagname=None):
    if filename is None or tagname is None:
        raise ValueError('File name and tag name should be set.')
    with h5py.File(os.path.join(dirname, filename), 'r') as hdf:
        return hdf[tagname][:]


def set_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
