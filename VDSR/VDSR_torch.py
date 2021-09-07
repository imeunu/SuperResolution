import os

import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class VDSR(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv = 

def get_dataset(args):
    with h5py.File(args['data_path'],'r') as f:
        lr = (np.array(f['lr']) - 127.5) / 255.0
        residual = np.array(f['residual'])
    return lr, residual

# def 


if __name__ == '__main__':
    args = {
        'data_path' : 'C:/Users/eunwoo/Desktop/Code/savehere/train_291.h5',
        'save_path' : 'C:/Users/eunwoo/Desktop/Code/savehere',
        'epoch' : 80,
        'momentum' : 0.9,
        'l2' : 1e-4,
        'lr' : 0.1,
        'batch_size' : 64,
        'grad_clip' : 0.4,
        'training' : True,
        'val_path' : 'C:/Users/eunwoo/Desktop/Images/Set5/bird_GT.bmp'}

    if args['training']:
        lr, residual = get_dataset(args)
        # train(lr, residual, args)

    # if args['val_path']:
    #     evaluate(args)
