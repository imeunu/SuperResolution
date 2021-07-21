from google.colab import drive
drive.mount('/content/gdrive')

import os

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model, model_from_json 
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from functions import plot, postprocess, psnr, save_model

with h5py.File('/content/gdrive/MyDrive/91-image_x2.h5','r') as f:
    print("Keys: %s" % f.keys())
    group_key = list(f.keys())
    hr = np.array(f['hr'])
    lr = np.array(f['lr'])
lr = np.expand_dims(lr,axis=-1)
hr = np.expand_dims(hr,axis=-1)

def modcrop(img, scale):
    tmpsz = img.shape
    sz = tmpsz[0:2]
    sz = sz - np.mod(sz, scale)
    img = img[0:sz[0], 1:sz[1]]
    return img

def shave(image, border):
    img = image[border: -border, border: -border]
    return img

def get_lr(path,factor):
    img = cv2.imread(path)
    size = int(len(img)/factor)
    lr = cv2.GaussianBlur(img,(5,5),0)
    lr = cv2.resize(lr,(size,size),cv2.INTER_AREA)
    lr = cv2.resize(lr,(len(img),len(img)),cv2.INTER_CUBIC)
    return lr

# Build SRCNN
def SRCNN():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size = (9, 9), activation='relu', padding='same'))
    model.add(Conv2D(filters=32, kernel_size = (1, 1), activation='relu', padding='same'))
    model.add(Conv2D(filters=1, kernel_size = (5, 5), activation='linear', padding='same'))
    adam = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_squared_error'])
    return model

save_path = '/content/gdrive/MyDrive/savehere'
callbacks_list = [
    ModelCheckpoint(filepath = save_path+"/weights.{loss:.4f}.hdf5",
                    monitor = 'loss', save_best_only=True),
    ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, mode='min', min_delta=1e-4)]
model = SRCNN()
history = model.fit(lr,hr,epochs=10000,batch_size=64,callbacks=callbacks_list)
save_model(model, save_path)
    
# Validation
test = cv2.imread(os.listdir()[1])
low = get_lr(test,4)
low = np.expand_dims(lr,axis=-1)
pred = np.squeeze(model.predict(low))

print('PSNR: ',psnr(test,pred))
from skimage.measure import compare_ssim as ssim
print('SSIM: ',ssim(test,pred,multichannel =True))
