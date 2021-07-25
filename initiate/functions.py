import numpy as np

def plot(img):
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def postprocess(img):
    img[img[:] > 255] = 255
    img[img[:] < 0] = 0
    return img

def psnr(x, y, peak=255):
    '''
    x: images
    y: another images
    peak: MAX_i peak. if int8 -> peak =255
    return: psnr value
    '''
    max_ = peak
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    diff = (x-y).flatten('C')
    mse = np.mean(diff**2)
    result = 10 * np.log10(_max/rmse)
    return result

def save_model(model, save_path):
    model_json = model.to_json()
    with open(save_path+"/model.json".format(acc), 'w') as json_file:
        json_file.write(model_json)

    model.save_weights(save_path +"/final_weight.h5")
    model_json = model.to_json()
    with open(save_path+"/model.json".format(acc), 'w') as json_file:
        json_file.write(model_json)

def rgb2ycbcr(img):
    '''Convert RGB numpy array into YCbCr numpy array'''
    r = img[:, :, 0]; g = img[:, :, 1]; b = img[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.5 * b
    cr = 0.5 * r - 0.419 * g - 0.081 * b
    return np.array([y,cb,cr])

def ycbcr2rgb(img):
    y = img[:, :, 0]; cb = img[:, :, 1]; cr = img[:, :, 2]
    r = y + 1.5748 * cr
    g = y - 0.18732 * cb - 0.46812 * cr
    b = y + 1.8556 * cb
    return np.array([r,g,b])
