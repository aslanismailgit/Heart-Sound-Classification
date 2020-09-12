#%%
import numpy as np
import json
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
import pandas as pd
from functions import *
# import sys
# del sys.modules['functions']
# from functions import *
#%%
# try:
#     x_train
#     print("Already imported")
# except:
#      print ("starting to load the data")
#      CLASS_LABELS, x_train, y_train, ind_train, \
#         x_val, y_val, ind_val, x_test,y_test, ind_test = import_data(dataset_no=1, add_dim=0)
#%%
def run_model_2d(dataset_no=1):
    CLASS_LABELS, x_train, y_train, ind_train, \
        x_val, y_val, ind_val, x_test,y_test, ind_test = import_data(dataset_no=1, add_dim=0)

    #%%
    Normal, Abnormal = np.bincount(y_train)
    total = Normal + Abnormal
    ratio = Normal / Abnormal
    print (f'Normal to Abnormal ratio: {ratio:.2f}')
    print('Examples:\n    Total: {}\n    Abnormal: {} ({:.2f}% of total)\n'.format(
        total, Abnormal, 100 * Abnormal / total))

    weight_for_0 = 1
    weight_for_1 = ratio
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    #%%
    tf.keras.backend.clear_session()

    #%%
    conv2d_input = keras.Input(shape=(50, 200,1), name="conv_2d")
    ksize = 7
    x = keras.layers.Conv2D(32, (ksize,ksize), activation="relu")(conv2d_input)
    x = keras.layers.MaxPooling2D(2, 2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Conv2D(64, (ksize,ksize), activation="relu")(x)
    x = keras.layers.MaxPooling2D(2, 2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Conv2D(128, (ksize,ksize), activation="relu")(x)
    x = keras.layers.MaxPooling2D(2, 2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    conv2d_output = keras.layers.GlobalAveragePooling2D()(x)
    # x = keras.layers.Dense(128, activation='relu')(x)
    # x = keras.layers.Dense(64, activation='relu')(x)
    # x = keras.layers.Dense(32, activation='relu')(x)
    # conv2d_output = keras.layers.Dense(10, activation='relu')(x)

    model_2d = keras.Model(inputs=conv2d_input, outputs=conv2d_output, name="conv2d_model")
    model_2d.summary()

    #%%
    c = keras.layers.Dense(256, activation="relu")(conv2d_output)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.Dropout(0.2)(c)

    c = keras.layers.Dense(128, activation="relu")(c)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.Dropout(0.2)(c)

    c = keras.layers.Dense(64, activation="relu")(c)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.Dropout(0.2)(c)

    c = keras.layers.Dense(32, activation="relu")(c)
    result = keras.layers.Dense(1,activation="sigmoid")(c) 
    model = keras.models.Model(inputs = conv2d_input ,outputs = result)
    model.summary()
    #%%
    callbacks, metrics = callbacks_metrics()

    model.compile(optimizer=tf.keras.optimizers.RMSprop(1e-4), 
                loss='binary_crossentropy', 
                metrics=metrics)
    model.summary()
    #%%
    epochs = 100
    history = model.fit( x_train, y_train, 
                        epochs=epochs, 
                        validation_data=(x_val,y_val),
                        verbose=1,
                        shuffle=True,
                        callbacks = [callbacks],
                        class_weight=class_weight)
    #%%

    recordPiece, fname = get_recordPiece()

    shortDisc = "conv2d ksize 7 Flatten No dense after convs "
    y_pred_prob_test = model.predict([x_test])
    modelname = ""
    filename = os.path.basename(__file__)
    DataSource = "Test"
    performance_te, temp_te, FNlist_te, FPlist_te, TPlist_te, TNlist_te = get_results (DataSource, shortDisc, model, y_pred_prob_test,y_test,ind_test,recordPiece, fname, modelname,filename)

    #%%
    _id = time.strftime("_%m_%d_%H")
    modelname =  './models/model' + _id + performance_te

    history_df = pd.DataFrame(history.history)
    history_df.head()
    history_df.to_csv("./models/history" + _id + performance_te + ".csv")
    model.save(modelname)
    #%%
    y_pred_prob_val = model.predict([x_val])
    DataSource = "Val"
    performance_val, temp_val, FNlist_val, FPlist_val, TPlist_val, TNlist_val = get_results (DataSource, shortDisc, model, y_pred_prob_val,y_val,ind_val,recordPiece, fname, modelname,filename)

    #%%
    y_pred_prob_train = model.predict(x_train)
    DataSource = "Train"
    performance_tr, temp_tr, FNlist_tr, FPlist_tr, TPlist_tr, TNlist_tr = get_results (DataSource, shortDisc, model, y_pred_prob_train,y_train, ind_train,recordPiece, fname, modelname,filename)

# #%%
# df = pd.read_csv("results.csv")
# pd.set_option("display.precision", 3)
# df.iloc[:,[0,4,5,6,7]]

# #%%
# import matplotlib.pyplot as plt
# acc = history.history['binary_accuracy']
# val_acc = history.history['val_binary_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'b.', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'b.', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

#%%====================================
