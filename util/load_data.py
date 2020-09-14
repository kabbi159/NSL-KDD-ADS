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


def set_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)


def get_hdf5_data(dirpath):
    hdf5_files = os.listdir(dirpath)
    print(dirpath)
    data = []
    label = []
    for hdf5_file in hdf5_files:
        with h5py.File(dirpath+'/'+hdf5_file,'r') as f:
            data.append(f['data'].value)
            label.append(f['label'].value)
        print(hdf5_file)
    return np.concatenate(data), np.concatenate(label).flatten()
