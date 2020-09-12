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
import pandas as pd
from functions import *
import sys
del sys.modules['functions']
from functions import *
#%%
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

#%%
ksize = 9
conv1d_input_1 = keras.Input(shape=(50, 200))
x = layers.Reshape((10000,1))(conv1d_input_1)
x = layers.Conv1D(filters=32, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPool1D(strides=2)(x)
x = keras.layers.Dropout(0.25)(x)

x = layers.Conv1D(filters=64, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPool1D(strides=2)(x)
x = keras.layers.Dropout(0.25)(x)

x = layers.Conv1D(filters=128, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPool1D(strides=2)(x)
x = keras.layers.Dropout(0.25)(x)

x = layers.Conv1D(filters=128, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPool1D(strides=2)(x)
x = keras.layers.Dropout(0.25)(x)

x = layers.Conv1D(filters=256, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPool1D(strides=2)(x)
x = keras.layers.Dropout(0.25)(x)

x = layers.Conv1D(filters=512, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPool1D(strides=2)(x)
x = keras.layers.Dropout(0.25)(x)

x = layers.Conv1D(filters=512, kernel_size=ksize, activation='relu',
                kernel_regularizer = l2(0.025))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPool1D(strides=2)(x)
x = keras.layers.Dropout(0.25)(x)

conv1d_output_1 = keras.layers.Flatten()(x)

# conv1d_output_1 = keras.layers.GlobalAveragePooling1D()(x)
model_1d_1 = keras.Model(inputs=conv1d_input_1, outputs=conv1d_output_1, name="conv1d_model_1")
model_1d_1.summary()

#%%
c = layers.Dense(256, activation="relu")(conv1d_output_1)
c = keras.layers.BatchNormalization()(c)
c = keras.layers.Dropout(0.5)(c)

# c = layers.Dense(256, activation="relu")(c)
# c = keras.layers.BatchNormalization()(c)
# c = keras.layers.Dropout(0.5)(c)

c = layers.Dense(128, activation="relu")(c)
c = keras.layers.BatchNormalization()(c)
c = keras.layers.Dropout(0.5)(c)

c = layers.Dense(64, activation="relu")(c)
c = keras.layers.BatchNormalization()(c)
c = keras.layers.Dropout(0.5)(c)

c = layers.Dense(32, activation="relu")(c)
c = keras.layers.Dropout(0.5)(c)

c = layers.Dense(16, activation="relu")(c)

result = keras.layers.Dense(1,activation="sigmoid")(c) 
model = keras.models.Model(inputs = conv1d_input_1 ,outputs = result)
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

shortDisc = "10000x1 conv1d ksize 9 Flatten No dense after convs "
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