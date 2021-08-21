import glob
import os

import cv2
import h5py
import numpy as np

def get_residual(img, scale):
    height, width, _ = img.shape
    lr = cv2.resize(img,(height//scale,width//scale), cv2.INTER_CUBIC)
    lr = cv2.resize(lr,(height,width),cv2.INTER_CUBIC)
    residual = img.split()[0] - lr.split()[0]
    return residual

def generate(args):
    try: os.mkdir(args['save_path'])
    except: pass
    h5 = h5py.File(args['save_path'],'w')
    lows, highs, scales = [], [], [2,3,4]

    for path in glob.glob('*.{}'.format(args['img_ext'])):
        hr = cv2.imread(path, cv2.IMREAD_COLOR).astype('float')
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2YCrCb)
        y = cv2.split(hr)[0]
        for scale in scales:
            residual = get_residual(hr,scale)
            lows.append(residual)
            highs.append(y)
    h5.create_dataset('residual', data = lows)
    h5.create_dataset('hr', data = highs)
    h5.close()

if __name__ == '__main__':
    args = {
        'img_dir' : 'C:\Users\eunwoo\Desktop\Code\srcnn\Train',
        'save_path' : 'C:\Users\eunwoo\Desktop\Code\savehere',
        'img_ext' : 'bmp',
        'size' : 41}
    generate(args)