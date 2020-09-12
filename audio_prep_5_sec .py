#%%
import json
import os
import librosa
import math
import numpy as np
import pandas as pd
import wave

#%%
def get_class(ref_df, filename):
    filename = filename.split(".")[0]
    val = ref_df[1][ref_df[0]==filename].values[0]
    if val==0:
        class_ = "Uncertain"
        labelNum = 2
    elif val==1:
        class_ = "Abnormal"
        labelNum = 1
    elif val == -1:
        class_ = "Normal"
        labelNum = 0
    return class_, labelNum, filename

#%%
DATASET_PATH = "C:/Users/ism/Documents/audio_data/medData/physionetData/train"
# JSON_PATH = DATASET_PATH + "/signal5sec.json"
winLength = 0.10 # sec
maxLen = 5 #sec
data1 = { "labels": [], "sigArray":[]}
data2 = {"mapping": [],"recordPiece":[],"pieceSeg":[],"fname":[]}

# Normal (-1)	Uncertain (0)	Abnormal (1
#%%
counter = 0 #2641
recordPieceArray = []
pieceSegArray = []
fnameArray = []
sigArray = []
labelArray=[]
mappingArray=[]
for i, (dirpath, dirnames, filenames) in enumerate(sorted(os.walk(DATASET_PATH))):
    #print(i, (dirpath, dirnames, filenames))

    if (dirpath is not DATASET_PATH):
        ref_file_path = dirpath + "/REFERENCE.csv"
        ref_df = pd.read_csv(ref_file_path,header=None)
        for f in filenames:
            print(counter)
            counter +=1
            if (f[0] != ".") and (f[0] != "R"):
                label, labelNum, fname = get_class(ref_df, f)
                if labelNum!=2:
                    file_path = os.path.join(dirpath, f)
                    with wave.open(file_path, "rb") as wave_file:    
                        SAMPLE_RATE = wave_file.getframerate()
                                    
                    signal, sample_rate = librosa.load(file_path,
                                                        sr=SAMPLE_RATE)
                    signal_length = signal.shape[0]
                    pieceLen = (sample_rate * maxLen)
                    pieceCount = int(signal_length / pieceLen )
 
                    for pieceCounter in range(pieceCount):
                        #print(f'Piece {pieceCounter} in progress')
                        pieceStart = pieceCounter * pieceLen
                        pieceEnds =  pieceStart + pieceLen
                        recordPiece = signal[pieceStart: pieceEnds]
                        
                        winLenSample = int(SAMPLE_RATE * winLength) # win len in terms of signal rate                     
                        num_of_signals = int((pieceLen) /(winLenSample))
                        sig_array_temp = np.empty((num_of_signals,winLenSample))
                        for sig in range(num_of_signals):
                            start = winLenSample * sig
                            ends = start + winLenSample
                            signal_temp = recordPiece[start:ends]
                            sig_array_temp[sig,:] = signal_temp

                            pieceSegArray.append(sig)
                        recordPieceArray.append(pieceCounter)

                        fnameArray.append(fname)
                        labelArray.append(labelNum)
                        mappingArray.append(label)
                        sigArray.append(sig_array_temp.tolist())

#%% ========== create 10 shuffled train, val, test data sets ========== 
number_of_sets=10
l = len(sigArray)
ind = np.arange(len(sigArray))

for i in range(number_of_sets):
    print(50*"-",i)

    train_size = int(.75*l)
    val_size = int(.15*l)
    test_size = l - (train_size + val_size)
    np.random.shuffle(ind)

    ind_train = ind[:train_size]
    x_train = np.array(sigArray)[ind_train]
    y_train = np.array(labelArray)[ind_train]

    ind_val = ind[train_size:-test_size]
    x_val = np.array(sigArray)[ind_val]
    y_val = np.array(labelArray)[ind_val]

    ind_test = ind[-test_size:]
    x_test = np.array(sigArray)[ind_test]
    y_test = np.array(labelArray)[ind_test]

    print("---------- TRAIN SET ----------")
    print(f'x_train shape: {(x_train.shape)},    {type(x_train)}')
    print(f'y_train shape: {(y_train.shape)}', {type(y_train)})
    print(f'ind_train shape: {(ind_train.shape)}, {type(ind_train)}')

    print("---------- VAL SET ----------")
    print(f'x_val shape: {(x_val.shape)},    {type(x_val)}')
    print(f'y_val shape: {(y_val.shape)}', {type(y_val)})
    print(f'ind_val shape: {(ind_val.shape)}, {type(ind_val)}')

    print("---------- TEST SET ----------")
    print(f'x_test shape: {(x_test.shape)},    {type(x_test)}')
    print(f'y_test shape: {(y_test.shape)}', {type(y_test)})
    print(f'ind_test shape: {(ind_test.shape)}, {type(ind_test)}')

    set_path = DATASET_PATH + "sets/set_" + str(i)

    if not os.path.exists(set_path):
        os.makedirs(set_path)

    PATH_train = set_path + "/signal5secTrain.json"
    PATH_val = set_path + "/signal5secVal.json"
    PATH_test = set_path + "/signal5secTest.json"


    data_train = { "y_train": [], "x_train":[], "ind_train":[]}
    data_train["x_train"]=x_train.tolist()
    data_train["y_train"]=y_train.tolist()
    data_train["ind_train"]=ind_train.tolist()
    # print(PATH_train)

    with open(PATH_train, "w") as fp:
        json.dump(data_train, fp, indent=4)

    data_val = { "y_val": [], "x_val":[], "ind_val":[]}
    data_val["x_val"]=x_val.tolist()
    data_val["y_val"]=y_val.tolist()
    data_val["ind_val"]=ind_val.tolist()
    # print(PATH_val)

    with open(PATH_val, "w") as fp:
        json.dump(data_val, fp, indent=4)

    data_test = { "y_test": [], "x_test":[], "ind_test":[]}
    data_test["x_test"]=x_test.tolist()
    data_test["y_test"]=y_test.tolist()
    data_test["ind_test"]=ind_test.tolist()
    # print(PATH_test)

    with open(PATH_test, "w") as fp:
        json.dump(data_test, fp, indent=4)

