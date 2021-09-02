import argparse
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,
                                kernel_size=9,padding='same')
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=32,
                                kernel_size=1,padding='same')
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=1,
                                kernel_size=5,padding='same')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class Trainset(Dataset):
    def __init__(self, h5_file):
        super(Trainset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            hr = np.array(f['hr'])
            lr = np.array(f['lr'])
            hr = torch.FloatTensor(hr).permute(0,3,1,2)
            lr = torch.FloatTensor(lr).permute(0,3,1,2)
            return lr[idx], hr[idx]

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

def train(args):
    train_data = Trainset(args.train_file)
    trainloader = DataLoader(dataset=train_data,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True,
                                  drop_last=True)
    model = SRCNN().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=[
                {"params": model.conv1.parameters()}, 
                {"params": model.conv2.parameters()}, 
                {"params": model.conv3.parameters(), 
                 "lr": args.lr * 0.1}], lr=args.lr)

    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            low, high = data
            low = low.to(device)
            high = high.to(device)

            optimizer.zero_grad()
            outputs = model(low)
            loss = criterion(outputs, high)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2 == 1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2))
                running_loss = 0.0

    print('Training Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--train_file', type = str, required = True)
    parser.add_argument('--batch_size', type=int, default=4, help='Number of epochs to train.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--save_pth', action='store_true', default=False, help='Save the checkpoint or not')

    args = parser.parse_args()
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:0' if args.cuda else 'cpu')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train(args)
