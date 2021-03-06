# 107368009jungchengtsai
作法說明

    train.data分成x與y的資料，y資料保留price，x資料則把id與price給剃除
    valid.data分成x與y的資料，y資料保留price，x資料則把id與price給剃除
    test.data分成x資料，x資料剃除id
    train_x與train_y，valid_x與valid_y做model訓練
    test_x資料讀進model做prediction
    
流程圖
![image](https://github.com/107368009jungchengtsai/107368009jungchengtsai/blob/master/process%20chart.png)
 
程式流程

    1.宣告和定義
   
    import pandas as pd
    import numpy as np
    from keras.models import Sequential
    from keras.layers import *
    from keras.callbacks import *
    from sklearn.preprocessing import *
    
    2.讀檔(x為輸入,y為輸出)
    
    df_train = pd.read_csv('train-v3.csv')
    df_y_train=df_train['price']
    df_x_train=df_train.drop(['id', 'price'], axis=1)
    
    3.確認model是否被訓練到
    
    df_valid = pd.read_csv('valid-v3.csv')
    df_y_valid=df_valid['price']
    df_x_valid=df_valid.drop(['id', 'price'], axis=1)
    
    4.老師給的資料丟進model預測出price
    
    df_test = pd.read_csv('test-v3.csv')
    df_x_test=df_test.drop(['id'], axis=1)
    
    5.正規化
    
    mean = df_x_train.mean(axis=0)
    df_x_train -= mean
    std = df_x_train.std(axis=0)
    df_x_train /= std
    df_x_valid -= mean
    df_x_valid /= std
    df_x_test -= mean
    df_x_test /= std
    
    6.訓練出來的值乘上權重
    
    def build_model():
    model = Sequential()
    model.add(Dense(256, activation='relu',input_shape=(df_x_train.shape[1],)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
    
    7.查看model
    
    model = build_model()
    model.summary()
    
    8.設定跑的次數，和程式跑的過程
    
    history = model.fit(df_x_train, df_y_train, validation_data=(df_x_valid,df_y_valid), epochs=1000)
![image](https://github.com/107368009jungchengtsai/107368009jungchengtsai/blob/master/process.png)
    
    9.看MAE圖表
    
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
 ![image](https://github.com/107368009jungchengtsai/107368009jungchengtsai/blob/master/MAE.png)
    
    
    10.test丟進model預測出price,結果存進csv檔案
    
    id_col = df_test['id']
    prediction = model.predict(df_x_test)
    prediction = prediction.astype(int)
    submission = pd.DataFrame()
    submission['id'] = id_col
    submission['price'] = prediction
    submission.to_csv('submission.csv', index=False)
   
Kaggle排名
![image](https://github.com/107368009jungchengtsai/107368009jungchengtsai/blob/master/kaggle.png)

分析

    1.epochs設越高並不會越好，這一次設超過1000就會overfitting
    2.層數設越多並不會變好
    
    
改進
   
    1.使用過correlation matrix，把關聯性低的資料給剃除
    2.尋找更好的層數
