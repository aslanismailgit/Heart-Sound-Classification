#%%
import numpy as np
import json
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from tensorflow.keras.regularizers import l2
# from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix
# from sklearn.metrics import accuracy_score, f1_score 
import pandas as pd

#%%
from functions import *
import sys
del sys.modules['functions']
from functions import *
try:
    x_train
    print("Already imported")
except:
     print ("starting to load data")
     CLASS_LABELS, x_train, y_train, ind_train, \
        x_val, y_val, ind_val, x_test,y_test, ind_test = import_data(add_dim=0)

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

print(f'Weight for class 0: {weight_for_0:.2f}')
print(f'Weight for class 1: {weight_for_1:.2f}')

#%%
tf.keras.backend.clear_session()
# tf.random.set_seed(51)
# np.random.seed(51)
try:
    del model
    print("model deleted")
except:
    pass

#%%
ksize = 9
filt_size = 64
conv1d_input_1 = keras.Input(shape=(50, 200))
x = layers.Reshape((10000,1))(conv1d_input_1)
# ===================================================
x = layers.Conv1D(filters=filt_size, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025))(x)
x = layers.Conv1D(filters=filt_size, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPool1D(strides=2)(x)
block_1_output = keras.layers.Dropout(0.25)(x)
# =================   (None, 4992, 32) ==============
ksize = 3
filt_size = 64

x = layers.Conv1D(filters=filt_size, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025), padding="same")(block_1_output)
x = layers.Conv1D(filters=filt_size, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025), padding="same")(x)
x = keras.layers.BatchNormalization()(x)
# x = keras.layers.MaxPool1D(strides=2)(x)
x = keras.layers.Dropout(0.25)(x)
block_2_output = layers.add([x, block_1_output], name="res----2")

x = layers.Conv1D(filters=filt_size, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025), padding="same")(block_2_output)
x = layers.Conv1D(filters=filt_size, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025), padding="same")(x)
x = keras.layers.BatchNormalization()(x)
# x = keras.layers.MaxPool1D(strides=2)(x)
x = keras.layers.Dropout(0.25)(x)
block_3_output = layers.add([x, block_2_output],name="res----3")

x = layers.Conv1D(filters=filt_size, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025), padding="same")(block_3_output)
x = layers.Conv1D(filters=filt_size, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025), padding="same")(x)
x = keras.layers.BatchNormalization()(x)
# x = keras.layers.MaxPool1D(strides=2)(x)
x = keras.layers.Dropout(0.25)(x)
block_4_output = layers.add([x, block_3_output],name="res----4")

x = layers.Conv1D(filters=filt_size, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025), padding="same")(block_4_output)
x = layers.Conv1D(filters=filt_size, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025), padding="same")(x)
x = keras.layers.BatchNormalization()(x)
# x = keras.layers.MaxPool1D(strides=2)(x)
x = keras.layers.Dropout(0.25)(x)
block_5_output = layers.add([x, block_4_output],name="res----5")

# =================   increase conv dim  ==============
ksize = 3
filt_size = 128
x = layers.Conv1D(filters=filt_size, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025))(block_5_output)
x = layers.Conv1D(filters=filt_size, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPool1D(strides=2)(x)
block_6_output = keras.layers.Dropout(0.25, name="res----6")(x)
# ================= (None, 2494, 128)     ==============
ksize = 3
filt_size = 128
x = layers.Conv1D(filters=filt_size, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025), padding="same")(block_6_output)
x = layers.Conv1D(filters=filt_size, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025), padding="same")(x)
x = keras.layers.BatchNormalization()(x)
# x = keras.layers.MaxPool1D(strides=2)(x)
x = keras.layers.Dropout(0.25)(x)
block_7_output = layers.add([x, block_6_output], name="res----7")

x = layers.Conv1D(filters=filt_size, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025), padding="same")(block_7_output)
x = layers.Conv1D(filters=filt_size, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025), padding="same")(x)
x = keras.layers.BatchNormalization()(x)
# x = keras.layers.MaxPool1D(strides=2)(x)
x = keras.layers.Dropout(0.25)(x)
block_8_output = layers.add([x, block_7_output], name="res----8")

x = layers.Conv1D(filters=filt_size, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025), padding="same")(block_8_output)
x = layers.Conv1D(filters=filt_size, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025), padding="same")(x)
x = keras.layers.BatchNormalization()(x)
# x = keras.layers.MaxPool1D(strides=2)(x)
x = keras.layers.Dropout(0.25)(x)
block_9_output = layers.add([x, block_8_output], name="res----9")

x = layers.Conv1D(filters=filt_size, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025), padding="same")(block_9_output)
x = layers.Conv1D(filters=filt_size, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025), padding="same")(x)
x = keras.layers.BatchNormalization()(x)
# x = keras.layers.MaxPool1D(strides=2)(x)
x = keras.layers.Dropout(0.25)(x)
block_10_output = layers.add([x, block_9_output], name="res----10")


#%%

x = layers.Conv1D(64, 3, activation="relu")(block_10_output)
x = layers.GlobalAveragePooling1D()(x)
# x = layers.Dense(512, activation="relu")(x)
# x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1,activation="sigmoid")(x)

model = keras.Model(conv1d_input_1, outputs, name="audio_resnet")
model.summary()


#%%
from functions import *
import sys
del sys.modules['functions']
from functions import *
callbacks, metrics = callbacks_metrics()
# %%
model.compile(optimizer=tf.keras.optimizers.RMSprop(1e-4), 
            loss='binary_crossentropy', 
            metrics=metrics)
model.summary()

#%%
epochs = 100
history = model.fit( [x_train], y_train, 
                    epochs=epochs,
                    batch_size=20,
                    validation_data=([x_val],y_val),
                    verbose=1,
                    shuffle=True,
                    callbacks = [callbacks],
                    class_weight=class_weight)


#%%
import matplotlib.pyplot as plt
acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'b.', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b.', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#%%====================================
