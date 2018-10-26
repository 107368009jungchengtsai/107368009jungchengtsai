# 107368009jungchengtsai
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
    
    9.看loss圖表與MAE圖表
    
    import matplotlib.pyplot as plt
    def plot_hist(h, xsize=6, ysize=10):
    fig_size = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = [xsize, ysize]
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True)
    
    plt.subplot(211)
    plt.plot(h['mean_absolute_error'])
    plt.plot(h['val_mean_absolute_error'])
    plt.title('Training vs Validation MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(212)
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('Training vs Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.draw()
    plt.show()
    return
    
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
