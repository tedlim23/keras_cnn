import sys
sys.path.append('D:\\experimental\\cutout\\src')
import os
import json
from tensorflow import keras as K
import numpy as np
import preprocess.models as models 
import dpath

SAVED = "src\\saved"
IMAGE_SIZE = (224, 224)

def load_train_result():
    if os.path.exists(os.path.join(SAVED, "result.json")):
        with open(os.path.join(SAVED, "result.json"), "r") as jf:
            train_result = json.load(jf)
            jf.close()
        
        max_fold = train_result['fold']
        dataname = train_result['dataname']
        print(f'Successfully read train result from fold {max_fold}!')
        weight_path = os.path.join(SAVED, f"model_fold_{max_fold}.h5")
    else:
        # absolute path for model if dbg
        dataname = "fruit"
        weight_path = SAVED+f"\\model_fold_1.h5"
    
    return dataname, weight_path

def decode_pred(val):
    probs = val[0]
    ind = np.argmax(probs)
    return ind, probs[ind]

def get_metrics(confmat):
    sum_axis0 = np.sum(confmat, axis=0)
    sum_axis1 = np.sum(confmat, axis=1)
    total_sum =  np.sum(confmat)
    diagonal = confmat.diagonal()
    tpsum = np.sum(diagonal)

    tpvec = diagonal
    fpvec = sum_axis0 - tpvec
    fnvec = sum_axis1 - tpvec
    tnvec = total_sum - tpvec - fpvec - fnvec
    tpfpfntn_vec = np.stack((tpvec,fpvec,fnvec,tnvec), axis=0)

    tpfpsum_vec = (tpvec+fpvec)
    tpfnsum_vec = (tpvec+fnvec)

    def zero_checking(vec):
        zero_where = np.where(vec==0)[0]
        if len(zero_where) > 0:
            idx = zero_where
            zero_flag = True
        else:
            idx = None
            zero_flag = False
        
        return zero_flag, idx

    precision_vec = np.true_divide(tpvec,tpfpsum_vec,where=(tpfpsum_vec!=0))
    zflag1, zidx1 = zero_checking(tpfpsum_vec)
    if zflag1:
        precision_vec[zidx1] = np.nan

    recall_vec = np.true_divide(tpvec,tpfnsum_vec,where=(tpfnsum_vec!=0))
    zflag2, zidx2 = zero_checking(tpfnsum_vec)
    if zflag2:
        recall_vec[zidx2] = np.nan
    
    if np.sum(np.isnan(recall_vec))>0:
        mean_recall = np.nanmean(recall_vec)
        recall_vec[np.isnan(recall_vec)] = 0.0
    else:
        mean_recall = np.sum(recall_vec)/len(recall_vec)
    
    if np.sum(np.isnan(precision_vec))>0:
        mean_prec = np.nanmean(precision_vec)
        precision_vec[np.isnan(precision_vec)] = 0.0
    else:
        mean_prec = np.sum(precision_vec)/len(precision_vec)
    
    acc = tpsum / total_sum

    return acc, mean_recall, mean_prec

def main():
    dataname, weight_path = load_train_result()
    # model = models.get_mbv2()
    model = models.get_resnet50()
    model = models.compile_model(model, "sgd", "binary")
    model.load_weights(weight_path)

    num_classes = 2
    path_dict = {
        'fruit': [dpath.fruit_newdir, dpath.fresh_dir_test, dpath.rot_dir_test],
        'inscape': [dpath.inscape_newdir, dpath.ok_dir_test, dpath.ng_dir_test],
        'dworld': [dpath.dworld_newdir, dpath.ok_dir_test, dpath.ng_dir_test]
    }
    imglist = []
    for i in range(num_classes):
        rootdir = os.path.join(path_dict[dataname][0], path_dict[dataname][i+1])
        filelist = os.listdir(rootdir)
        fullpath = [os.path.join(rootdir,f) for f in filelist]
        imglist.append(fullpath)
    
    confmat = np.zeros(shape=(num_classes, num_classes), dtype=np.int32)

    for clsidx, filelist in enumerate(imglist):
        for j, img_path in enumerate(filelist):
            if j > 0 and j % 50 == 0:
                print(f'{j}/{len(filelist)} is complete for class {clsidx}')
                sys.stdout.write("\033[F")
            img = K.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
            x = K.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = K.applications.resnet.preprocess_input(x)
            preds = model.predict(x)
            pred_cls, prob = decode_pred(preds)
            np.add.at(confmat, (clsidx, pred_cls), 1)
    print("\nInference Complete")

    print(f"confmat: \n", confmat)
    acc, mean_recall, mean_prec = get_metrics(confmat)
    print("acc: ", acc)
    print("mean_recall: ", mean_recall)
    print("mean_prec: ", mean_prec)