import glob
import os

import cv2
import h5py

from functions import walk

def get_residual(img, scale):
    height, width, _ = img.shape
    lr = cv2.resize(img,(height//scale,width//scale), cv2.INTER_CUBIC)
    lr = cv2.resize(lr,(height,width),cv2.INTER_CUBIC)
    residual = img.split()[0] - lr.split()[0]
    return residual

def generate(args):
    try: os.mkdir(args['save_path'])
    except: pass
    h5 = h5py.File(os.path.join(args['save_path'],'train_291.h5'),'w')
    lows, highs, scales = [], [], [2,3,4]

    for folder, filename in walk(args['img_dir']):
        ext = os.path.splitext(filename)[-1]
        if not ext in args['img_ext']:
            continue
        path = os.path.join(folder,filename).replace(r'\\\\\\\\\\\\\\\','/')
        hr = cv2.imread(path, cv2.IMREAD_COLOR)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2YCrCb)
        y = cv2.split(hr)[0]
        for scale in scales:
            residual = get_residual(hr,scale)
            lows = sliceNappend(residual,lows,args)
            highs = sliceNappend(y,highs,args)
    h5.create_dataset('residual', data = lows)
    h5.create_dataset('hr', data = highs)
    h5.close()

def sliceNappend(img,imgs,args):
    for w in range(0, img.shape[0] - args['size'] + 1, args['size']):
        for h in range(0,img.shape[1] - args['size']+1):
            imgs.append(img[w : w + args['size'], h : h + args['size']])
    return imgs

if __name__ == '__main__':
    args = {
        'img_dir' : 'C:\Users\eunwoo\Desktop\Code\srcnn\Train',
        'save_path' : 'C:\Users\eunwoo\Desktop\Code\savehere',
        'img_ext' : 'bmp',
        'size' : 41}
    generate(args)
