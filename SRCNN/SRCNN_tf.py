import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers

from tensorflow.keras.models import Sequential, load_model, model_from_json 
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from functions import ycbcr2rgb, postprocess, psnr, save_model

def get_lr(img, scale, width, height, radius=5):
    '''Get Low Resolution PIL Image from High Resolution PIL Image'''
    w, h = img.width, img.height
    lr = hr.filter(ImageFilter.GaussianBlur(radius))
    lr = lr.resize((img.width // scale, img.height // scale),
                   resample = Image.BICUBIC)
    lr = lr.resize((width, height), resample = Image.BICUBIC)
    return lr

def SRCNN():
    model = models.Sequential()
    model.add(layers.Conv2D(filters = 64, kernel_size = (9, 9),
                     activation = 'relu', padding = 'same'))
    model.add(layers.Conv2D(filters = 32, kernel_size = (1, 1), 
                     activation = 'relu', padding = 'same'))
    model.add(layers.Conv2D(filters = 1, kernel_size = (5, 5), 
                     activation = 'linear', padding = 'same'))
    adam = optimizers.Adam(learning_rate = args.lr)

    model.compile(loss = 'mean_squared_error', 
                  optimizer = adam, 
                  metrics = ['mean_squared_error'])
    
    return model

def train(args):
    with h5py.File(args.data_path,'r') as f:
        hr = np.array(f['hr'])
        lr = np.array(f['lr'])
    lr = np.expand_dims(lr, axis = -1)
    hr = np.expand_dims(hr, axis = -1)

    callbacks_list = [
        ModelCheckpoint(filepath = args.save_path+"/weights.{loss:.4f}.hdf5",
                        monitor = 'loss', 
                        save_best_only = True),
        ReduceLROnPlateau(monitor = 'loss', 
                          factor = 0.1, 
                          patience = 10, 
                          verbose = 1, 
                          mode = 'min', 
                          min_delta = 1e-4)]
    model = SRCNN()
    history = model.fit(lr, hr, epochs = args.epochs, 
                        batch_size = args.batch_size, 
                        callbacks = callbacks_list)
    save_model(model, args.save_path)

    return model

def run(path,scale,model):
    img = Image.open(path)
    lr = get_lr(img,scale).convert('YCbCr')
    y, cb, cr = lr.split()
    y = np.expand_dims(np.array(y), axis = 0)
    y = np.expand_dims(y, axis = -1)
    y = np.squeeze(model.predict(y))
    pred = ycbcr2rgb(np.array([y,cb,cr]))
    pred = postprocess(pred)
    result = psnr(img,pred)
    print('predicted psnr:',result)
    print('lr psnr:',psnr(lr,img))
    # print('SSIM: ',ssim(test,pred,multichannel =True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, required = True)
    parser.add_argument('--save_path', type = str, required = True)
    parser.add_argument('--test_path', type = str, default = None)
    parser.add_argument('--scale', type = int, default = 3)
    parser.add_argument('--epochs', type = int, default = 1000)
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--lr', type = float, default = 1e-4)
    args = parser.parse_args()

    train(args)
    if not args.test_path:
        run(args.test_path, args.scale, model)
