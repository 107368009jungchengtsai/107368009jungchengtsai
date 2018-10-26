
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import keras as keras
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import *
from sklearn.preprocessing import *
df_train = pd.read_csv('train-v3.csv')
df_test = pd.read_csv('test-v3.csv')
df_x_train=df_train.drop(['id', 'price'], axis=1)
df_x_test=df_test.drop(['id'], axis=1)

mean = df_x_train.mean(axis=0)
std = df_x_train.std(axis=0)
df_x_test -= mean
df_x_test /= std
id_col = df_test['id']
model = keras.models.load_model("model.h5")
prediction = model.predict(df_x_test)
prediction = prediction.astype(int)
submission = pd.DataFrame()
submission['id'] = id_col
submission['price'] = prediction
submission
submission.to_csv('submission.csv', index=False)


# In[ ]:



