import argparse
import glob
import os

import h5py
import numpy as np
from PIL import Image, ImageFilter

from functions import rgb2y

def generate(args):
    save_path = os.path.join(args.save_path,'data.h5')
    h5 = h5py.File(save_path, 'w')
    highs, lows = [], []

    os.chdir(args.img_dir)
    for path in glob.glob('*.{}'.format(args.img_ext)):
        hr = Image.open(path).convert('RGB')
        lr = get_lr(hr, args.scale, args.radius)
        hr = rgb2y(np.array(hr).astype(np.float32))
        lr = rgb2y(np.array(lr).astype(np.float32))

        for w in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for h in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                highs.append(hr[w:w + args.patch_size, h:h + args.patch_size])
                lows.append(lr[w:w + args.patch_size, h:h + args.patch_size])

    lows = np.array(lows)
    highs = np.array(highs)
    print('Saved',np.shape(lows)[0],'Patches')

    h5.create_dataset('lr', data = lows)
    h5.create_dataset('hr', data = highs)
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
