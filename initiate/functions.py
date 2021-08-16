import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

def walk(folder):
    '''Walk through every files in a directory'''
    for dirpath, dirs, files in os.walk(folder):
        for filename in files:
            yield dirpath, filename

def plot(img):
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def postprocess(img):
    img[img[:] > 255] = 255
    img[img[:] < 0] = 0
    return img

def psnr(img1, img2):
    '''
    x: image
    y: another image
    return: psnr value
    '''
    img1 = img1.astype(np.float64) / 255.
    img2 = img2.astype(np.float64) / 255.
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return "Same Image"
    return 10 * np.log10(1. / mse)

def save_model(model, save_path):
    model_json = model.to_json()
    with open(save_path+"/model.json", 'w') as json_file:
        json_file.write(model_json)

    model.save_weights(save_path +"/final_weight.h5")
    model_json = model.to_json()
    with open(save_path+"/model.json", 'w') as json_file:
        json_file.write(model_json)

def rgb2y(img):
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:,:,0] + 129.057 * img[:,:,1] + 25.064 * img[:,:,2]) / 256.
        return np.array([y]).transpose([1,2,0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0,:,:] + 129.057 * img[1,:,:] + 25.064 * img[2,:,:]) / 256.
    else:
        raise TypeError('Unknown Type', type(img))

def rgb2ycbcr(img):
    if type(img) == np.ndarray:
        if len(np.shape(img)) > 3:
            img = np.squeeze(img)
        y = 16. + (64.738 * img[:,:,0] + 129.057 * img[:,:,1] + 25.064 * img[:,:,2]) / 256.
        cb = 128. + (-37.945 * img[:,:,0] - 74.494 * img[:,:,1] + 112.439 * img[:,:,2]) / 256.
        cr = 128. + (112.439 * img[:,:,0] - 94.154 * img[:,:,1] - 18.285 * img[:,:,2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0,:,:] + 129.057 * img[1,:,:] + 25.064 * img[2,:,:]) / 256.
        cb = 128. + (-37.945 * img[0,:,:] - 74.494 * img[1,:,:] + 112.439 * img[2,:,:]) / 256.
        cr = 128. + (112.439 * img[0,:,:] - 94.154 * img[1,:,:] - 18.285 * img[2,:,:]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise TypeError('Unknown Type', type(img))

def ycbcr2rgb(img):
    y = img[:,:,0]; cb = img[:,:,1]; cr = img[:,:,2]
    r = y + 1.5748 * cr
    g = y - 0.18732 * cb - 0.46812 * cr
    b = y + 1.8556 * cb
    return np.array([r,g,b])

def get_lr(img, scale, radius=5):
    '''Get Low Resolution PIL Image from High Resolution PIL Image'''
    lr = img.filter(ImageFilter.GaussianBlur(radius))
    lr = lr.resize((img.width // scale, img.height // scale),
                   resample = Image.BICUBIC)
    lr = lr.resize((img.width, img.height), resample = Image.BICUBIC)
    return lr
