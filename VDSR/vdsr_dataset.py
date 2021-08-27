import os

import cv2
import h5py

def walk(folder):
    for dirpath, dirs, files in os.walk(folder):
        for filename in files:
            yield dirpath, filename

def get_residual(img, scale):
    height, width, _ = img.shape
    lr = cv2.resize(img,(width//scale,height//scale), cv2.INTER_CUBIC)
    lr = cv2.resize(lr,(width,height),cv2.INTER_CUBIC)
    residual = cv2.split(img)[0] - cv2.split(lr)[0]
    return residual, lr

def generate(args):
    try: os.mkdir(args['save_path'])
    except: pass
    h5 = h5py.File(os.path.join(args['save_path'],'train_291.h5'),'w')
    lows, residuals, scales = [], [], [2,3,4]

    for folder, filename in walk(args['img_dir']):
        ext = os.path.splitext(filename)[-1]
        if not ext in args['img_ext']:
            continue
        path = os.path.join(folder,filename)
        hr = cv2.imread(path, cv2.IMREAD_COLOR)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2YCrCb)
        for scale in scales:
            residual, lr = get_residual(hr,scale)
            lows = sliceNappend(cv2.split(lr)[0],lows,args)
            residuals = sliceNappend(residual,residuals,args)
    h5.create_dataset('residual', data = residuals)
    h5.create_dataset('lr', data = lows)
    h5.close()

def sliceNappend(img,imgs,args):
    for w in range(0, img.shape[0] - args['size'] + 1, args['size']):
        for h in range(0,img.shape[1] - args['size']+1):
            imgs.append(img[w : w + args['size'], h : h + args['size']])
    return imgs

if __name__ == '__main__':
    args = {
        'img_dir' : 'C:\\Users\\eunwoo\\Desktop\\Images',
        'save_path' : 'C:\\Users\\eunwoo\\Desktop\\Code\\savehere',
        'img_ext' : ['.bmp','.jpg'],
        'size' : 41}
    generate(args)
