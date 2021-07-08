import numpy as np

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
    rmse = np.sqrt(np.mean(diff**2))
    result = 20 * np.log10(_max/rmse)
    return result
