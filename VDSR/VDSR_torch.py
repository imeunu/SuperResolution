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
        self.input_layer = nn.Conv2d(in_channels=1,out_channels=64,
                               kernel_size=3,stride=1,padding=1,bias=False)

        residual_block = []
        for _ in range(18):
            residual_block.append(nn.Conv2d(in_channels=64,out_channels=64,
                                  kernel_size=3,stride=1,padding=1,bias=False))
            residual_block.append(nn.RelU(inplace=True))
        self.residual_block = nn.Sequential(*residual_block)

        self.output_layer = nn.Conv2d(in_channels=64,out_channels=1,
                                      kernel_size=3,stride=1,
                                      padding=1,bias=False)

        self.weight_initialize()
    
    def forward(self,x):
        low_resolution = x
        output = self.input_layer(low_resolution)
        output = self.residual_block(output)
        output = self.output_layer(output)
        return output + low_resolution

    def weight_initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu')

class Trainset(Dataset):
    def __init__(self,h5_path):
        super(Trainset, self).__init__()
        self.h5_path = h5_path

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            lr = np.array(f['lr'])
            residual = np.array(f['residual'])
            lr = torch.FloatTensor(lr).permute(0,3,1,2)
            residual = torch.FloatTensor(residual).permute(0,3,1,2)
            return lr[idx], residual[idx]

    def __len__(self):
        with h5py.File(self.h5_path, 'r') as f:
            return len(f['lr'])

def get_dataset(args):
    with h5py.File(args['data_path'],'r') as f:
        lr = np.array(f['lr'])
        residual = np.array(f['residual'])
    return lr, residual

def train(args):
    train_data = get_dataset(args)
    # trainloader = 


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
        train(lr, residual, args)

    # if args['val_path']:
    #     evaluate(args)
