from google.colab import drive
drive.mount('/content/gdrive')
import os
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File('/content/gdrive/MyDrive/91-image_x2.h5','r') as f:
    print("Keys: %s" % f.keys())
    group_key = list(f.keys())
    hr = np.array(f['hr'])
    lr = np.array(f['lr'])
lr = np.expand_dims(lr,axis=-1)
hr = np.expand_dims(hr,axis=-1)

def plot(img):
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def modcrop(img, scale):
    tmpsz = img.shape
    sz = tmpsz[0:2]
    sz = sz - np.mod(sz, scale)
    img = img[0:sz[0], 1:sz[1]]
    return img

def shave(image, border):
    img = image[border: -border, border: -border]
    return img
  
import keras
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model, model_from_json 
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# Build SRCNN
def build_SRCNN():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size = (9, 9), activation='relu', padding='same'))
    model.add(Conv2D(filters=32, kernel_size = (1, 1), activation='relu', padding='same'))
    model.add(Conv2D(filters=1, kernel_size = (5, 5), activation='linear', padding='same'))
    adam = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_squared_error'])
    return model

save_path = '/content/gdrive/MyDrive/savehere'
callbacks_list = [
    ModelCheckpoint(filepath = save_path+"/weights.{loss:.4f}.hdf5",
                    monitor = 'loss', save_best_only=True),
    ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, mode='min', min_delta=1e-4)]
model = build_SRCNN()
history = model.fit(lr,hr,epochs=400,batch_size=64,callbacks=callbacks_list)
  
# Save Architecture
model_json = model.to_json()
output = model.predict(test_X)
predicted_classes = output.argmax(axis=1)
answer_classes = test_Y.argmax(axis=1)
acc = accuracy_score(answer_classes, predicted_classes)
with open(save_path+"/model_acc_{:.4f}.json".format(acc), 'w') as json_file:
    json_file.write(model_json)

# Save Weight
model.save_weights(save_path +"/final_weight.h5")
model_json = model.to_json()
with open(save_path+"/model.json".format(acc), 'w') as json_file:
    json_file.write(model_json)
    
test = cv2.imread(os.listdir()[1])
def get_lr(path,factor):
    img = cv2.imread(path)
    size = int(len(img)/factor)
    lr = cv2.GaussianBlur(img,(5,5),0)
    lr = cv2.resize(lr,(size,size),cv2.INTER_AREA)
    lr = cv2.resize(lr,(len(img),len(img)),cv2.INTER_CUBIC)
    return lr


# Validation
low = get_lr(os.listdir()[1],4)
low = np.expand_dims(lr,axis=-1)

pred = model.predict(lr)

def postprocess(pred):
  pred[pred[:] > 255] = 255
  pred[pred[:] < 0] = 0
  pred = pred.astype(np.uint8)
  return pred

def psnr(x, y, peak=255):
    '''
    :param x: images
    :param y: another images
    :param peak: MAX_i peak. if int8 -> peak =255
    :return: return psnr value
    '''
    _max = peak
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    diff = (x-y).flatten('C')
    rmse = np.sqrt(np.mean(diff**2))
    result = 20 * np.log10(_max/rmse)
    return result
  
print(psnr(test,test))
