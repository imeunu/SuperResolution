import os

import cv2
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint

from functions import postprocess
from vdsr_dataset import get_residual

def get_dataset(args):
    with h5py.File(args['data_path'],'r') as f:
        lr = (np.array(f['lr']) - 127.5) / 255.0
        residual = np.array(f['residual'])
    return lr, residual

def VDSR(args,data_num):
    batch_num = data_num // args['batch_size'] + 1
    model = models.Sequential()
    model.add(layers.Conv2D(
              filters = 64, kernel_size = (3,3),
              activation = 'relu', padding = 'same',
              kernel_initializer = 'glorot_uniform',
              kernel_regularizer = regularizers.l2(args['l2']),
              input_shape = (None,None,1)))
    for i in range(18):
        model.add(layers.Conv2D(
                  filters = 64, kernel_size = (3,3),
                  activation = 'relu', padding = 'same',
                  kernel_initializer = 'glorot_uniform',
                  kernel_regularizer = regularizers.l2(args['l2'])))
    model.add(layers.Conv2D(
              filters = 1, kernel_size = (3,3),
              activation = 'linear', padding = 'same',
              kernel_initializer = 'glorot_uniform',
              kernel_regularizer = regularizers.l2(args['l2'])))
    
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate = args['lr'],
        decay_steps = batch_num * 20,
        decay_rate = 0.1)

    sgd = optimizers.SGD(learning_rate = lr_schedule,
                         momentum = args['momentum'],
                         clipnorm = args['grad_clip'])

    model.compile(loss = 'mean_squared_error',
                  optimizer = sgd,
                  metrics = ['mean_squared_error'])
    return model

def train(lr, residual, args):
    lr = np.expand_dims(lr,axis = -1)
    residual = np.expand_dims(residual, axis = -1)
    callbacks_list = [
        ModelCheckpoint(
                    filepath = args['save_path'] + "/weights.{loss:.4f}.hdf5",
                    monitor = 'loss', 
                    save_best_only = True)]
    model = VDSR(args,len(lr))
    history = model.fit(lr, residual, epochs = args['epochs'],
                        batch_size = args['batch_size'],
                        callbacks = callbacks_list)
    # save_history(history,args['save_path'])

def load_model(jsonpath,weightpath):
    with open(jsonpath) as f:
        json = f.read()
    model = tf.keras.models.model_from_json(json)
    model.load_weights(weightpath)
    return model

def evaluate(args):
    json = os.path.join(args['save_path'], 'model.json')
    weight = os.path.join(args['save_path'], 'final_weight.h5')
    model = load_model(json,weight)

    img = cv2.imread(args['val_path'], cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    _, lr = get_residual(img)
    y, cr, cb = cv2.split(lr)
    y = (y - 127.5) / 255.

    residual = model.predict(y)
    y = postprocess(y + residual)
    ycrcb = cv2.merge(y, cr, cb)
    predicted = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    print('PSNR:', cv2.PSNR(img,predicted))
    return predicted

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

    if args['val_path']:
        evaluate(args)
