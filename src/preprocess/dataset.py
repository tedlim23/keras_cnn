import os
import cv2
import numpy as np
import tensorflow as tf
import preprocess.aug as aug

def resize_to_dir(rootdir, subdir, dstdir, ext):
    ext = "."+ext
    olddir = os.path.join(rootdir, subdir)
    imgset = []
    for root, dirs, files in os.walk(olddir):
        for j, filename in enumerate(files):
            if ext in filename:
                full_path = os.path.join(root, filename)
                imgset.append(full_path)
    print("imgset N : ", len(imgset))
    rsize = (224,224)
    newdir = os.path.join(dstdir, subdir)
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    
    for imgpath in imgset:
        try:
            img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            fname = os.path.basename(imgpath)
            fname_png = fname.split(ext)[0]+".png"
            resized = cv2.resize(img, rsize, interpolation = cv2.INTER_LINEAR)
            newpath = os.path.join(newdir, fname_png)
            cv2.imwrite(newpath, resized)
        except Exception as ex:
            print(f'Unable to resize : {ex}')

def read_imgpath(x_dir, ext = "png"):
    num_classes = len(x_dir)
    x_all = []
    ext = "."+ext
    class_cnts = [0 for i in range(num_classes)]
    for i, x in enumerate(x_dir):
        for root, dirs, files in os.walk(x):
            for filename in files:
                if ext in filename:
                    full_path = os.path.join(root, filename)
                    x_all.append(full_path)
                    class_cnts[i] += 1

    X = np.asarray(x_all)
    label_idx = [i for i in range(num_classes)]
    Y = np.repeat(label_idx, class_cnts)
    # import pdb; pdb.set_trace()

    np.random.seed(21)
    np.random.shuffle(X)
    np.random.seed(21)
    np.random.shuffle(Y)

    return X, Y

def onehot_encode(Y):
    from sklearn.preprocessing import OneHotEncoder
    Y = Y.reshape(-1,1)
    enc = OneHotEncoder()
    enc.fit(Y)
    Y_1hot = enc.transform(Y).toarray()
    Y_1hot = Y_1hot.astype(np.uint8)
    return Y_1hot

def dbg_img(img):
    # img = img/255
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img2 = img2.astype(np.uint8)
    cv2.imshow('win-dbg', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def path2data(imglist, labels, batch_size):
    num_classes = len(labels[0])
    imgset = [[] for i in range(num_classes)]
    labelset = [[] for i in range(num_classes)]
    for i, imgpath in enumerate(imglist):
        image = tf.keras.preprocessing.image.load_img(
            imgpath, grayscale=False, color_mode="rgb", target_size=None, interpolation="linear"
        )
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        # input_arr = tf.keras.applications.mobilenet_v2.preprocess_input(input_arr)
        input_arr = tf.keras.applications.resnet.preprocess_input(input_arr)
        input_arr = aug.cutout(input_arr)
        # dbg_img(input_arr)
        
        onehot = labels[i]
        ind = np.argmax(onehot)
        imgset[ind].append(input_arr)
        labelset[ind].append(onehot) 
    

    imgset = np.asarray(imgset)
    labelset = np.asarray(labelset)
    
    dataset = []
    total_len = 0
    for j, imglist in enumerate(imgset):
        cls_img_len = len(imglist)
        img_dataset = tf.data.Dataset.from_tensor_slices((imglist, labelset[j]))
        img_dataset = img_dataset.repeat()
        img_dataset = img_dataset.shuffle(buffer_size = cls_img_len)
        total_len += cls_img_len
        dataset.append(img_dataset)
    print("TF datasets have been loaded successfully")
    print("Total Train Length :", total_len)

    choice_dataset =  tf.data.Dataset.range(num_classes).repeat()
    choice_dataset = choice_dataset.shuffle(buffer_size = total_len)
    input_dataset = tf.data.experimental.choose_from_datasets(dataset, choice_dataset)
    input_dataset = input_dataset.batch(batch_size)
    input_dataset = input_dataset.prefetch(1)

    return input_dataset

