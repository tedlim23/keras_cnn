import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import random

def cutout(img):
    if len(img.shape) == 3:
        h, w = img.shape[0], img.shape[1]
        img = np.expand_dims(img, axis = 0)
        
        maskh = 84
        maskw = 84
        offh = random.randint(0, h)
        offw = random.randint(0, w)
    elif len(img.shape) == 4:
        h, w = img.shape[1], img.shape[2]
        maskh = 84
        maskw = 84
        offh = random.randint(0, h)
        offw = random.randint(0, w)
    
    img_ret = tfa.image.cutout(
        images = img,
        mask_size = (maskh, maskw),
        offset = (offh, offw),
        constant_values= 0,
        data_format= 'channels_last'
    )
    # img_ret = np.squeeze(img, axis = 0)
    return img_ret

def hflip(img):
    return tf.image.flip_left_right(img)

def vflip(img):
    return tf.image.flip_up_down(img)

def brightness(img):
    return tf.image.random_brightness(
        image, 0.5, seed=None
    )

def contrast(img):
    return tf.image.random_contrast(
        img, 0.1, 1.5, seed=None
    )

def hue(img):
    return tf.image.random_hue(
        image, 0.25, seed=None
    )

def crop(img):
    BATCH_SIZE = 1
    NUM_BOXES = 3
    IMAGE_HEIGHT = 224
    IMAGE_WIDTH = 224
    CHANNELS = 3
    CROP_SIZE = (224, 224)

    image = np.expand_dims(img, axis = 0)
    boxes = tf.random.uniform(shape=(NUM_BOXES, 4))
    box_indices = tf.random.uniform(shape=(NUM_BOXES,), minval=0, maxval=BATCH_SIZE, dtype=tf.int32)
    output = tf.image.crop_and_resize(image, boxes, box_indices, CROP_SIZE)
    # output.shape  #=> (5, 24, 24, 3)
    return output

if __name__ == "__main__":
    import cv2
    filepath = r"D:\experimental\cutout\data\tmp\_206_4262.png"
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img_ret = cutout(img)
    # img2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_ret = img_ret.astype(np.uint8)
    cv2.imshow('win-dbg', img_ret)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
