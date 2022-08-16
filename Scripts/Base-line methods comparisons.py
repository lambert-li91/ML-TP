
from google.colab import drive
from google.colab import files

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from scipy.fftpack import fft,ifft
from scipy.optimize import curve_fit
import datetime
from sklearn.preprocessing import MinMaxScaler
from scipy import signal
import time
from scipy.spatial.distance import cdist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from keras.models import load_model 
from sklearn.metrics import r2_score
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import tensorflow as tf



def LSTMm3(generator, n_features =1, time_step =3, epochs=150):
    from numpy.random import seed
    tf.random.set_seed(2)
    model = Sequential()   
    model.add(LSTM(5, activation='relu', input_shape=(time_step, n_features), return_sequences=True))
    model.add(LSTM(5, activation='relu', return_sequences=True))
    model.add(LSTM(1, activation='relu'))
    model.compile(optimizer='adam', loss='mae')    
    model.fit_generator(generator,  epochs =epochs, shuffle=False, verbose=0)
    wt1 = model.get_weights()

    return model, wt1



#%%
def LSTMmt(data3, sid ,time_step, epochs):
    ctt = data3[data3.sqID==sid].copy()  
    ctt.drop(['sqID'],axis=1,inplace=True)
    ctt.reset_index(drop=True, inplace=True)
    min_max_scaler = MinMaxScaler()
   
    temp_traffic = min_max_scaler.fit_transform(np.array(ctt['traffic']).reshape(-1,1))
    ctt['traffic'] = temp_traffic.reshape(-1)
    ctt_train_X, ctt_test_X, ctt_train_y, ctt_test_y = ctt.values[0:lentrain],ctt.values[168*5:],ctt['traffic'].values[0:lentrain],ctt['traffic'].values[168*5:]
    ctt_generator_train = TimeseriesGenerator(ctt_train_X, ctt_train_y, length=time_step, batch_size=batch_size, shuffle=False)
    ctt_generator_test = TimeseriesGenerator(ctt_test_X, ctt_test_y, length=time_step, batch_size=batch_size, shuffle=False)
    cttmodel, wt = LSTMm3(ctt_generator_train, n_features =3, time_step =time_step, epochs = epochs)
    ctt_y_hat = cttmodel.predict_generator(ctt_generator_test)

    ctt_y_true = ctt_generator_test.targets[time_step:]    
    ctt_y_hat = ctt_y_hat.reshape(-1)
    ctt_y_true = ctt_y_true.reshape(-1)

    tempMSE = mean_squared_error(ctt_y_hat, ctt_y_true)
    tempRMSE = np.sqrt(tempMSE)
    tempR2 = r2_score(ctt_y_hat, ctt_y_true)
    tempMAE = mean_absolute_error(ctt_y_hat, ctt_y_true)
    error = ctt_y_hat - ctt_y_true
    tempAbsMAE = np.abs(ctt_y_hat - ctt_y_true)
  
    return wt, cttmodel, tempMSE, tempRMSE, tempR2, tempMAE


drive.mount('/content/gdrive')
!ls
import os
os.chdir("/content/gdrive/My Drive/Milan")


filename2 = r'preprocessed_data_mid7wks_indiv_standardlized.csv'
data4 = pd.read_csv(filename2)
order = ['sqID','hour','wkday','traffic']
data4 = data4[order]


time_step = 3
lentrain = 168*1
epochs = 150
batch_size = 8
ind = np.load('ind.npy')
testID = 7121

weights, model, temp_MSE, temp_RMSE, temp_R2, temp_MAE = LSTMmt(data4, testID, time_step, epochs)




from statsmodels.tsa.arima_model import ARIMA

def arimaPredict(data3,group):
  for sid in group:
    print('Training ARIMA...')
    print(sid)

    ctt = data3[data3.sqID==int(sid)].copy() 
    ctt.drop(['sqID'],axis=1,inplace=True)
        
    ctt.reset_index(drop=True, inplace=True)
    arima_train, arima_test = ctt['traffic'].values[0:lentrain],ctt['traffic'].values[168*5:]
    history = [x for x in arima_train]
    predictions = list()

    
    for t in range(len(arima_test)):    
      model = ARIMA(history, order=(3,1,0))
      model_fit = model.fit(disp=0)
      output = model_fit.forecast()
      yhat = output[0]
      predictions.append(yhat)
      obs = arima_test[t]
      history.append(obs)
    predictions = np.array(predictions).reshape(-1)


    plt.figure()
    plt.tick_params(labelsize=15)
    plt.plot(arima_test,label = 'Ground truth')
    plt.plot(predictions,label = 'Predictions')
    plt.xlabel('Time (Hour)',font1)
    plt.ylabel('Traffic',font1)
    plt.legend(prop = fontlabel)

    
    error = predictions - arima_test
    plt.figure()
    plt.tick_params(labelsize=15)
    plt.bar(np.arange(0,len(error)),error)
    #plt.plot(error)
    plt.xlabel('Time (Hour)',font1)
    plt.ylabel('Error',font1)
  
    tempMSE = mean_squared_error(predictions, arima_test)
    tempRMSE = np.sqrt(tempMSE)
    tempR2 = r2_score(predictions, arima_test)
    tempMAE = mean_absolute_error(predictions, arima_test)

    print('MSE: %f'%tempMSE)
    print('RMSE: %f'%tempRMSE)
    print('MAE: %f'%tempMAE)
    print('R2: %f'%tempR2)
    

arimaPredict(df_1,PAdf)



def LRPredict(data3,group):
  for sid in group:
    print('Training LR...')
    print(sid)

    ctt = data3[data3.sqID==int(sid)].copy() 
    ctt.drop(['sqID'],axis=1,inplace=True)
    ctt.reset_index(drop=True, inplace=True)
    X_train, X_test = ctt['traffic'].values[0:lentrain],ctt['traffic'].values[168*5:]
    predictions = list()

    y_train = X_train
    y_test = X_test

    X_train = np.arange(0,len(y_train))
    X_test = np.arange(len(y_train),(len(y_train)+len(y_test)))
    

    poly_feature = PolynomialFeatures(degree = 19)
    X_train_2 = poly_feature.fit_transform(X_train.reshape(-1,1))
    X_test_2 = poly_feature.fit_transform(X_test.reshape(-1,1))

    history = [x for x in X_train]
    predictions = list()

    for t in range(len(X_test)):
      regressor_model=linear_model.LinearRegression()
      regressor_model.fit(X_train_2,y_train)

      y_hat = regressor_model.predict(X_test_2[t].reshape(1,-1))
      X_train_2 = np.append(X_train_2,X_test_2[t].reshape(1,-1),axis=0)
      y_train = np.append(y_train,np.array(y_test[t]).reshape(-1),axis=0)
      #y_train.append(y_test(t))
      
      predictions.append(y_hat)
    predictions = np.array(predictions).reshape(-1)


    plt.figure()
    plt.tick_params(labelsize=15)
    plt.plot(y_test,label = 'Ground truth')
    plt.plot(predictions,label = 'Predictions')
    plt.xlabel('Time (Hour)',font1)
    plt.ylabel('Traffic',font1)
    plt.legend(prop = fontlabel)
    
    error = predictions - y_test
    tempMSE = mean_squared_error(predictions, y_test)
    tempRMSE = np.sqrt(tempMSE)
    tempR2 = r2_score(predictions, y_test)
    tempMAE = mean_absolute_error(predictions, y_test)

    print('MSE: %f'%tempMSE)
    print('RMSE: %f'%tempRMSE)
    print('MAE: %f'%tempMAE)
    print('R2: %f'%tempR2)

LRPredict(df_1,PAdf)



def SVRPredict(data3,group):
  for sid in group:
    print('Training SVR...')
    print(sid)
    ctt = data3[data3.sqID==int(sid)].copy() 
    ctt.drop(['sqID'],axis=1,inplace=True)
    ctt.reset_index(drop=True, inplace=True)

    min_max_scaler = MinMaxScaler()
    temp_traffic = min_max_scaler.fit_transform(np.array(ctt['traffic']).reshape(-1,1))
    ctt['traffic'] = temp_traffic.reshape(-1)
    X_train, X_test, y_train, y_test = ctt.values[0:lentrain],ctt.values[lentrain:],ctt['traffic'].values[0:lentrain],ctt['traffic'].values[lentrain:]  
    predictions = list()

    X_train_2 = X_train
    X_test_2 = X_test

    regressor_model = SVR(kernel = 'rbf')

    history = [x for x in X_train]
    predictions = list()

    for t in range(len(X_test)):
      regressor_model.fit(X_train_2,y_train)

      y_hat = regressor_model.predict(X_test_2[t].reshape(1,-1))
      X_train_2 = np.append(X_train_2,X_test_2[t].reshape(1,-1),axis=0)
      y_train = np.append(y_train,np.array(y_test[t].reshape(-1)),axis=0)
      
      predictions.append(y_hat)
    predictions = np.array(predictions).reshape(-1)


    y_test = min_max_scaler.inverse_transform((y_test).reshape(-1,1)).reshape(-1)
    predictions = min_max_scaler.inverse_transform((predictions).reshape(-1,1)).reshape(-1)

    plt.figure()
    plt.tick_params(labelsize=15)
    plt.plot(y_test,label = 'Ground truth')
    plt.plot(predictions,label = 'Predictions')
    plt.xlabel('Time (Hour)',font1)
    plt.ylabel('Traffic',font1)
    plt.legend(prop = fontlabel)

    
    error = predictions - y_test
    plt.figure()
    plt.tick_params(labelsize=15)
    plt.bar(np.arange(0,len(error)),error)
    plt.xlabel('Time (Hour)',font1)
    plt.ylabel('Error',font1)
    
    print(predictions.shape)
    print(y_test.shape)
    tempMSE = mean_squared_error(predictions, y_test)
    tempRMSE = np.sqrt(tempMSE)
    tempR2 = r2_score(predictions, y_test)
    tempMAE = mean_absolute_error(predictions, y_test)

    print('MSE: %f'%tempMSE)
    print('RMSE: %f'%tempRMSE)
    print('R2: %f'%tempR2)
    print('MAE: %f'%tempMAE)
for runcounter in np.arange(0,1):

  poolA = list(tempind)

  PAdf = pd.DataFrame(data=None)
  cell_missing = []
  for i in poolA:
      temp = df[df.sqID==i].traffic
      if len(temp)!=1176:
        cell_missing.append(i)
        temp = np.lib.pad(temp,(0,1176-len(temp)),'constant',constant_values=(0, 0))
      PAdf['%d'%i] = np.array(temp)

SVRPredict3(data3,PAdf,lentrain=168)


