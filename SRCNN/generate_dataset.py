import argparse
import glob
import os

import h5py
import numpy as np
from PIL import Image, ImageFilter

def get_lr(img, scale, width, height, radius=5):
    '''Get Low Resolution PIL Image from High Resolution PIL Image'''
    w, h = img.width, img.height
    lr = hr.filter(ImageFilter.GaussianBlur(radius))
    lr = lr.resize((img.width // scale, img.height // scale),
                   resample = Image.BICUBIC)
    lr = lr.resize((width, height, reample = Image.BICUBIC))
    return lr

def generate(args):
    h5 = h5py.File(args.save_path, 'w')
    highs, lows = [], []

    for path in glob.glob('*.{}'.format(args.img_ext)):
        hr = Image.open(path).convert('RGB')
        hr_w = (hr.width // args.scale) * args.scale
        hr_h = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_w, hr_h), resample = Image.BICUBIC)
        lr = get_lr(hr, args.scale, hr_w, hr_h, args.radius)
        hr = convert_rgb_to_y(hr)
        lr = np.array(lr).astype(np.float32)
        lr = convert_rgb_to_y(lr)

        for w in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                highs.append(hr[i:i + args.patch_size, j:j + args.patch_size])
                lows.append(lr[i:i + args.patch_size, j:j + args.patch_size])

    lows = np.array(lows)
    highs = np.array(highs)

    h5_file.create_dataset('lr', data = lows)
    h5_file.create_dataset('hr', data = highs)
    h5.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type = str, required = True)
    parser.add_argument('--img_ext', type = str, default = 'bmp')
    parser.add_argument('--save_path', type = str, required = True)
    parser.add_argument('--patch_size', type = int, default = 33)
    parser.add_argument('--scale', type = int, default = 3)
    parser.add_argument('--stride', type = int, default = 14)
    parser.add_argument('--radius', type = int, default = 5)
    args = parser.parse_args()

    generate(args)
