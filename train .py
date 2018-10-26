
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import *
from sklearn.preprocessing import *
df_train = pd.read_csv('train-v3.csv')
df_y_train=df_train['price']
df_x_train=df_train.drop(['id', 'price'], axis=1)

df_valid = pd.read_csv('valid-v3.csv')
df_y_valid=df_valid['price']
df_x_valid=df_valid.drop(['id', 'price'], axis=1)
mean = df_x_train.mean(axis=0)
df_x_train -= mean
std = df_x_train.std(axis=0)
df_x_train /= std
df_x_valid -= mean
df_x_valid /= std
def build_model():
    model = Sequential()
    model.add(Dense(256, activation='relu',input_shape=(df_x_train.shape[1],)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
model = build_model()
model.summary()
history = model.fit(df_x_train, df_y_train, validation_data=(df_x_valid,df_y_valid), epochs=10)
import matplotlib.pyplot as plt
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('model.h5')
del model

