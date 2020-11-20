import tensorflow_addons as tfa
import numpy as np
import random

def cutout(img):
    h, w = img.shape[0], img.shape[1]
    img = np.expand_dims(img, axis = 0)
    
    maskh = 56
    maskw = 56
    offh = random.randint(0, h)
    offw = random.randint(0, w)
    
    img = tfa.image.cutout(
        images = img,
        mask_size = (maskh, maskw),
        offset = (offh, offw),
        constant_values= 0,
        data_format= 'channels_last'
    )
    img_ret = np.squeeze(img, axis = 0)
    return img_ret
