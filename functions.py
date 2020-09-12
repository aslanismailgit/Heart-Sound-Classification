import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score 

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping

def import_data(dataset_no=1, add_dim=1):
    
    # i=np.arange(10)
    # np.random.shuffle(i)
    # i=i[0]
    i = dataset_no # use same dataset for each model
    DATASET_PATH = "C:/Users/ismai/Documents/audio_data/medData/physionetData/train"
    set_path = DATASET_PATH + "sets/set_" + str(i)
    PATH_train = set_path + "/signal5secTrain.json"
    PATH_val = set_path + "/signal5secVal.json"
    PATH_test = set_path + "/signal5secTest.json"

  
    with open(PATH_train, "r") as fp:
        dataread = json.load(fp)
    
    x_train = np.array(dataread["x_train"])
    y_train = np.array(dataread["y_train"])
    ind_train = np.array(dataread["ind_train"])

    print(x_train.shape)
    print(y_train.shape)

    with open(PATH_val, "r") as fp:
        dataread = json.load(fp)
  
    x_val = np.array(dataread["x_val"])
    y_val = np.array(dataread["y_val"])
    ind_val = np.array(dataread["ind_val"])

    print(x_val.shape)
    print(y_val.shape)
    
  
    with open(PATH_test, "r") as fp:
        dataread = json.load(fp)
   
    x_test = np.array(dataread["x_test"])
    y_test = np.array(dataread["y_test"])
    ind_test = np.array(dataread["ind_test"])

    print(x_test.shape)
    print(y_test.shape)
    if add_dim == 1:
        x_train = x_train[...,np.newaxis]
        x_val = x_val[...,np.newaxis]
        x_test = x_test[...,np.newaxis]

    CLASS_LABELS = ["Normal", "Abnormal"]
    x_axis_labels = CLASS_LABELS
    y_axis_labels = CLASS_LABELS
    #{0 : "Normal", 1: "Abnormal"}

    return CLASS_LABELS, x_train, y_train, ind_train, x_val, y_val, ind_val, x_test,y_test,ind_test

def callbacks_metrics():
    #%%
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8**x)

    callback_early_stopping = EarlyStopping(monitor='loss',
                                            patience=10, verbose=1,
                                            restore_best_weights=True)

    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.1,
                                        min_lr=1e-8,
                                        patience=0,
                                        verbose=0)
    callbacks=[ annealer,callback_early_stopping,callback_reduce_lr]
    metrics = [
        keras.metrics.Accuracy(),
        keras.metrics.BinaryAccuracy(),
        # keras.metrics.Precision(),
        # keras.metrics.Recall(),
        # keras.metrics.SensitivityAtSpecificity(0.5),
        # keras.metrics.SpecificityAtSensitivity(0.5),
        # keras.metrics.AUC()
    ]
    return callbacks, metrics


def get_recordPiece():
    datapath_fnames = r"C:/Users/ismai/Documents/audio_data/medData/physionetData/train/3247namepiece5sec.json"
    with open(datapath_fnames, "r") as fp:
        datapath_fnames = json.load(fp)

    mapping = np.array(datapath_fnames["mapping"])
    recordPiece = np.array(datapath_fnames["recordPiece"])
    pieceSeg = np.array(datapath_fnames["pieceSeg"])
    fname = np.array(datapath_fnames["fname"])
    print("Data succesfully loaded!")
    print(mapping.shape)
    print(recordPiece.shape)
    print(pieceSeg.shape)
    print(fname.shape)
    return recordPiece, fname

# _ =eval_performance(model,x_test,y_test)

def eval_performance(model, x_train, y_train):
    loss, acc, pr, rec, se, sp, auc = model.evaluate(x_train, y_train, verbose=0)
    # print(loss, acc, pr, rec, se, sp, auc)
    #%%
    f1 =  2*((pr*rec)/(pr+rec))
    print ('--------------------')
    print(f'Sensitivity:  0.96 / {se:.3f} ')
    print(f'Specificity:  1    / {sp:.3f} ')
    print(f'f1         :  0.98 / {f1:.3f} ') 
    print(f'Accuracy: {acc:.3f} ')
    print(f'Recall: Detects {rec:.3f} of abnormals')
    print(f'Precision: It is %{pr:.3f} of the time correct')
    print(f'ROC Area under curve:  {auc:.3f}')
    print(f'Loss: {loss:.3f} ')

    return loss, acc, pr, rec, se, sp, auc


def get_results (DataSource, shortDisc, model, y_pred_prob, y_test,ind_test,recordPiece, fname,modelname, filename):
    y_pred = y_pred_prob > 0.5
    y_true = y_test
    ac = accuracy_score(y_true,y_pred)
    recal = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    auc = roc_auc_score (y_true, y_pred)
    f1 = f1_score (y_true, y_pred)
    conf_mat = confusion_matrix(y_true,y_pred)
    # a good classifier roc is colse to 1
    TN = conf_mat[0,0]
    FP = conf_mat[0,1]
    FN = conf_mat[1,0]
    TP = conf_mat[1,1]
    rec = TP / (TP+FN) #sensitivity
    TNR = TN / (TN + FP) # specificity = 1 - false alarm
    # yani yüksel olması iyi
    print (f'\n{DataSource} Set :')
    print ('--------------------')
    print(f'Sensitivity:  0.96 / {rec:.3f} ')
    print(f'Specificity:  1    / {TNR:.3f} ')
    print(f'f1         :  0.98 / {f1:.3f} ') 
    print(f'Accuracy: {ac:.3f} ')
    print(f'Recall: Detects {recal:.3f} of abnormals')
    print(f'Precision: It is %{prec:.3f} of the time correct')
    print(f'ROC Area under curve:  {auc:.3f}')

    print(f'Confusion Matrix:\n {conf_mat}')
    performance = f'_se_{rec:.3f}_sp_{TNR:.3f}_auc_{auc:.3f}'
    if DataSource == "Test":
        modelname =  './models/model' + performance
    # print(f'Saved Model:\n {modelname}')
    print(shortDisc)
# %
    y_pr = np.empty((y_test.shape[0]))
    def to10(s):
        return 1 if s > 0.5 else 0
    for i in range(y_test.shape[0]):
        y_pr[i] = to10(y_pred_prob[i,0])
#%
    fn=0
    fp=0
    tn=0
    tp=0
    FPlist = []
    FNlist = []
    TPlist = []
    TNlist = []
    for i,ind in enumerate(ind_test):
        if y_true[i]==1 and y_pr[i]==0:
            fn+=1
            n = fname[ind]+"_"+str(recordPiece[ind])
            FNlist.append(n)
        if y_true[i]==0 and y_pr[i]==1:
            fp+=1
            n = fname[ind]+"_"+str(recordPiece[ind])
            FPlist.append(n)
        if y_true[i]==0 and y_pr[i]==0:
            tn+=1
            n = fname[ind]+"_"+str(recordPiece[ind])
            TNlist.append(n)
        if y_true[i]==1 and y_pr[i]==1:
            tp+=1
            n = fname[ind]+"_"+str(recordPiece[ind])
            TPlist.append(n)

    print(tn,fp)
    print(fn,tp)
    # print(len(FNlist))
    # print(len(FPlist))
    FNlist_name = []
    for n in FNlist:
        t = n.split("_")[0]
        FNlist_name.append(t) 
    FPlist_name = []
    for n in FPlist:
        t = n.split("_")[0]
        FPlist_name.append(t) 
    TNlist_name = []
    for n in TNlist:
        t = n.split("_")[0]
        TNlist_name.append(t) 
    TPlist_name = []
    for n in TPlist:
        t = n.split("_")[0]
        TPlist_name.append(t) 
    fp_fn = set(FPlist_name) & set(FNlist_name)
    fp_tp = set(FPlist_name) & set(TPlist_name)
    fp_tn = set(FPlist_name) & set(TNlist_name)
    fn_tp = set(FNlist_name) & set(TPlist_name)
    fn_tn = set(FNlist_name) & set(TNlist_name)
    tn_tp = set(TNlist_name) & set(TPlist_name)

    temp = [DataSource, ac, rec, TNR, f1, recal,prec,auc, TN, 
            FP, FN, TP, filename, shortDisc, modelname]
    df = pd.read_csv("results.csv")
    df.loc[len(df)] = temp
    df.to_csv("results.csv", index=False)
    return performance, temp