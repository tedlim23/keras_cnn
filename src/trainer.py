import sys
sys.path.append('D:\\experimental\\cutout\\src')

import itertools
import os
import json

import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

import preprocess.dataset as dataset
import preprocess.models as models
import dpath 

print("TF version:", tf.__version__)
print("GPU available : ", tf.config.list_physical_devices('GPU'))
print("CUDA status: ",tf.test.is_built_with_cuda())

def count_class(y):
    num_classes = len(y[0])
    cnts = [0 for i in range(num_classes)]
    for onehot in y:
        onehot = onehot.tolist()
        ind = onehot.index(1)
        cnts[ind] += 1
    return cnts

def train_fold(X, Y, train_index, test_index, expr):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    print("y_train", count_class(y_train))
    print("y_test", count_class(y_test))

    num_classes = len(y_train[0])
    print(f'{num_classes} classes are found')
    # model = models.get_mbv2()
    model = models.get_resnet50(num_classes)
    # model = models.fine_tune(model, 99)
    model = models.compile_model(model, "sgd", "categorical")
    
    batch_size = 64
    trainset = dataset.path2data(X_train, y_train, batch_size, "train", expr)
    valset = dataset.path2data(X_test, y_test, batch_size, "test", expr)

    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_test) // batch_size
    num_epoch = 100
    hist = model.fit(
        trainset,
        epochs=num_epoch, steps_per_epoch=steps_per_epoch,
        validation_data=valset,
        validation_steps=validation_steps
    ).history

    return model, hist
    

def main(dataname, expr):
    path_dict = {
        ## dataname : [resize_dir, class1_dir, ... , classN_dir, original_dir, file_extension]
        'fruit': [dpath.fruit_newdir, dpath.fresh_dir_train, dpath.rot_dir_train, dpath.fruit_dir, "png"],
        'inscape': [dpath.inscape_newdir, dpath.ok_dir_train, dpath.ng_dir_train, dpath.inscape_dir, "bmp"],
        'dworld': [dpath.dworld_newdir, dpath.ok_dir_train, dpath.ng_dir_train, dpath.dworld_dir, "bmp"],
        'hanrim': [dpath.hanrim_newdir, dpath.ok_dir_train, dpath.ng_dir_train, dpath.hanrim_dir, "jpg"],
        'implant': [dpath.implant_newdir, dpath.implant_D_train, dpath.implant_I_train, dpath.implant_O_train, dpath.implant_dir, "png"],
        'autoever': [dpath.autoever_newdir, dpath.autoever_ok_train, dpath.autoever_p1_train, dpath.autoever_p2_train, dpath.autoever_dir, "png"],
        'chest': [dpath.chest_newdir, dpath.chest_ok_train, dpath.chest_ng_train, dpath.chest_dir, "jpeg"],
    }

    path_list = path_dict[dataname]
    X_dir = []
    for subpath in path_list[1:-2]:
        target_dir = os.path.join(path_list[0], subpath)
        if not os.path.exists(target_dir):
            dataset.resize_to_dir(path_list[-2], subpath, path_list[0], ext = path_list[-1])
        X_dir.append(target_dir)
    
    X, Y = dataset.read_imgpath(X_dir)
    Y = dataset.onehot_encode(Y)

    scores = []
    kf = KFold(n_splits=3)
    kf.get_n_splits(X)
    for k, (train_index, test_index) in enumerate(kf.split(X)):
        print("# of train: ", len(train_index))
        print("# of test: ", len(test_index))

        model, modelhist = train_fold(X, Y, train_index, test_index, expr)
        modelname = f'model_fold_{k}.h5'
        model.save_weights(f"src/saved/{modelname}")
        
        losses = modelhist['val_loss']
        losses.sort()
        min5 = losses[:5]
        scores.append(float(np.mean(min5)))
    
    min_score = min(scores)
    with open("src/saved/result.json", "w+") as jf:
        myd = {
            'dataname': dataname,
            'score': min_score,
            'fold': scores.index(min_score)
        }
        json.dump(myd, jf)
        jf.close()

    print("scores : ", scores)