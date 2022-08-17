
from google.colab import drive
from google.colab import files


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from scipy import signal
from sklearn import neighbors
from sklearn.model_selection  import cross_val_score
from keras.models import load_model 
from keras.preprocessing.sequence import TimeseriesGenerator

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import MinMaxScaler
import time
import tensorflow as tf
from sklearn.metrics import r2_score



drive.mount('/content/gdrive')
!ls
import os
os.chdir("/content/gdrive/My Drive/Milan")



def dfs(xn, N):
    n = np.arange(0,N).reshape(-1,1)
    k = np.arange(0,N).reshape(-1,1)
    WN = np.exp(-1j*2*np.pi/N)
    nk = n.T*k
    WNnk = pow(WN,nk)
    Xk = np.dot(xn,WNnk)
    Xk2 = Xk[[1,7,14,21,28]]
    energy = np.sum(np.abs(Xk))
    return Xk2
  
  
def calDataframeDFS(df):
    FreqArray = np.array([])
    n = 0
    for i in df.columns:
        y1 = signal.detrend(np.array(df[i]))
        temp = y1.T[:168]
        tdfs = dfs(temp,len(temp))

        treal = tdfs.real.reshape(1,-1).copy()
        timag = tdfs.imag.reshape(1,-1).copy()
        tflatten = np.concatenate((treal ,timag),axis=1)
        if n==0:
            FreqArray = tflatten.copy()
            n = n+1
        else:
            FreqArray = np.concatenate((FreqArray ,tflatten),axis=0)
    return FreqArray
  
  def initModel2(n_features =1, time_step =2):
    tf.random.set_seed(2)
    model = Sequential()
    model.add(LSTM(5, activation='relu', input_shape=(time_step, n_features), return_sequences=True))
    model.add(LSTM(5, activation='relu', return_sequences=True))
    model.add(LSTM(1, activation='relu'))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    return model


def LSTMm3(generator, n_features =1, time_step =3, epochs=150):
    tf.random.set_seed(2)
    model = Sequential()   
    model.add(LSTM(5, activation='relu', input_shape=(time_step, n_features), return_sequences=True))
    model.add(LSTM(5, activation='relu', return_sequences=True))
    model.add(LSTM(1, activation='relu'))
    model.compile(optimizer='adam', loss='mae', metrics=['mse'])
    
    model.set_weights(ref_weights)
    print('Training...')
    
    model.fit_generator(generator,  epochs =epochs, shuffle=False, verbose=0, validation_data=generator_test)
    wt1 = model.get_weights()

    return model, wt1

#For base learner
def LSTMm3_bl(generator, initialWeight, n_features =1, time_step =3, epochs=150):
    tf.random.set_seed(2)
    model = Sequential()   
    model.add(LSTM(5, activation='relu', input_shape=(time_step, n_features), return_sequences=True))
    model.add(LSTM(5, activation='relu', return_sequences=True))
    model.add(LSTM(1, activation='relu'))
    model.compile(optimizer='adam', loss='mae', metrics=['mse'])
    
    model.set_weights(initialWeight)
    print('Base learner updating...')
    
    model.fit_generator(generator,  epochs =epochs, shuffle=False, verbose=0, validation_data=generator_test)
    wt1 = model.get_weights()

    return model, wt1


def convertWeight(a):
    t=[]
    for i in np.arange(len(a)):
        temp = a[i].flatten()
        t.extend(temp)
    t = np.array(t)
    return t

def setweights(model1, model2, model3):
     
    t = []
    
    wtlist1 = convertWeight(model1.get_weights())
    wtlist2 = convertWeight(model2.get_weights())
    wtlist3 = convertWeight(model3.get_weights())
    wtlist = (wtlist1 + wtlist2 +wtlist3)/3

    wl1 = wtlist[0:3*20].reshape(3,20)
    wl2 = wtlist[3*20:3*20+5*20].reshape(5,20)
    wl3 = wtlist[3*20+5*20:3*20+5*20+20].reshape(20,-1).flatten()
    wl4 = wtlist[3*20+5*20+20:3*20+5*20+20+5*20].reshape(5,20)
    wl5 = wtlist[3*20+5*20+20+5*20:3*20+5*20+20+5*20+5*20].reshape(5,20)
    wl6 = wtlist[3*20+5*20+20+5*20+5*20:3*20+5*20+20+5*20+5*20+20].reshape(20,-1).flatten()
    wl7 = wtlist[3*20+5*20+20+5*20+5*20+20:3*20+5*20+20+5*20+5*20+20+5*4].reshape(5,4)
    wl8 = wtlist[3*20+5*20+20+5*20+5*20+20+5*4:3*20+5*20+20+5*20+5*20+20+5*4+1*4].reshape(1,4)
    wl9 = wtlist[3*20+5*20+20+5*20+5*20+20+5*4+1*4:].reshape(1,-1).flatten()

    t.append(wl1)
    t.append(wl2)
    t.append(wl3)
    t.append(wl4)
    t.append(wl5)
    t.append(wl6)
    t.append(wl7)
    t.append(wl8)
    t.append(wl9)
    
    return t




#%%
def LSTMmt(data3, sid ,initBLweights, time_step, epochs):
    print('Training data length: %d'%lentrain)
    #%Extract 1 Cell's traffic
    ctt = data3[data3.sqID==sid].copy()  
    ctt.drop(['sqID'],axis=1,inplace=True)
    ctt.reset_index(drop=True, inplace=True)
    
    #print(ctt)
    min_max_scaler = MinMaxScaler()
    #print('shape:')
    #print(ctt.traffic.shape)
    
    temp_traffic = min_max_scaler.fit_transform(np.array(ctt['traffic']).reshape(-1,1))
    ctt['traffic'] = temp_traffic.reshape(-1)

    ctt_train_X, ctt_test_X, ctt_train_y, ctt_test_y = ctt.values[0:lentrain],ctt.values[840:840+168*2],ctt['traffic'].values[0:lentrain],ctt['traffic'].values[840:840+168*2]
    ctt_generator_train = TimeseriesGenerator(ctt_train_X, ctt_train_y, length=time_step, batch_size=batch_size, shuffle=False)
    ctt_generator_test = TimeseriesGenerator(ctt_test_X, ctt_test_y, length=time_step, batch_size=batch_size, shuffle=False)

    cttmodel, wt = LSTMm3(ctt_generator_train, n_features =3, time_step =time_step, epochs = epochs)
    blmodel, wt = LSTMm3_bl(ctt_generator_train, initBLweights, n_features =3, time_step =time_step, epochs = epochs)

    ctt_y_hat = cttmodel.predict(ctt_generator_test)
    bl_y_hat = blmodel.predict(ctt_generator_test)

    temp_ctt_y_hat = min_max_scaler.inverse_transform(np.array(ctt_y_hat).reshape(-1,1))
    ctt_y_hat = temp_ctt_y_hat.reshape(-1)
    temp_bl_y_hat = min_max_scaler.inverse_transform(np.array(bl_y_hat).reshape(-1,1))
    bl_y_hat = temp_bl_y_hat.reshape(-1)

    temp_ctt_y_true = min_max_scaler.inverse_transform((np.array(ctt_generator_test.targets[time_step:])).reshape(-1,1))
    ctt_y_true = temp_ctt_y_true.reshape(-1)
    temp_bl_y_true = min_max_scaler.inverse_transform((np.array(ctt_generator_test.targets[time_step:])).reshape(-1,1))
    bl_y_true = temp_bl_y_true.reshape(-1)

    tempMSE = mean_squared_error(ctt_y_hat, ctt_y_true)
    print('cal MSE: %f'%tempMSE)
    print('auto MSE: %f'%mean_squared_error(ctt_y_hat,ctt_y_true))
    tempRMSE = np.sqrt(tempMSE)
    tempR2 = r2_score(ctt_y_hat, ctt_y_true)
    tempMAE = mean_absolute_error(ctt_y_hat, ctt_y_true)

    tempMSE_bl = mean_squared_error(bl_y_hat, bl_y_true)
    tempRMSE_bl = np.sqrt(tempMSE_bl)
    tempR2_bl = r2_score(bl_y_hat, bl_y_true)

    plt.figure()
    plt.tick_params(labelsize=15)
    plt.plot(ctt_y_true,'k-',label = 'Ground truth',linewidth=2.5)
    plt.plot(ctt_y_hat,'r-',label = 'Stacked LSTM',linewidth=2.5)
    plt.xlabel('Time (Hour)',font1)
    plt.ylabel('Traffic',font1)
    plt.legend(prop = fontlabel)

    plt.figure()
    plt.tick_params(labelsize=15)
    plt.plot(bl_y_true,'k-',label = 'Ground truth',linewidth=2.5)
    plt.plot(bl_y_hat,'r-',label = 'MLCTPF',linewidth=2.5)
    plt.xlabel('Time (Hour)',font1)
    plt.ylabel('Traffic',font1)
    plt.legend(prop = fontlabel)

    error = ctt_y_hat - ctt_y_true
    plt.figure()
    plt.tick_params(labelsize=15)
    plt.bar(np.arange(0,len(error)),error)
    plt.xlabel('Time (Hour)',font1)
    plt.ylabel('Error',font1)

    error_bl = bl_y_hat - bl_y_true
    plt.figure()
    plt.tick_params(labelsize=15)
    plt.bar(np.arange(0,len(error_bl)),error_bl)
    plt.xlabel('Time (Hour)',font1)
    plt.ylabel('Error',font1)

    tempAbsMAE = np.abs(ctt_y_hat - ctt_y_true)
    tempAbsMAE_plot = pd.Series(tempAbsMAE)

    tempAbsMAE_bl = np.abs(bl_y_hat - bl_y_true)
    tempAbsMAE_bl_plot = pd.Series(tempAbsMAE_bl)

    return wt, cttmodel, tempMSE, tempRMSE, tempR2, tempMAE, tempMSE_bl, tempRMSE_bl, tempR2_bl, tempMAE_bl



def calFreqDistV2(groupFreqA, groupFreqB):
  FreqDist = []
  for i in np.arange(len(groupFreqA)):
    for j in np.arange(len(groupFreqB)):
      dist = [np.linalg.norm(groupFreqA[i] - groupFreqB[j])]
      FreqDist.extend(dist)
  return FreqDist

filename2 = r'preprocessed_data_mid7wks_indiv_standardlized.csv'
data3 = pd.read_csv(filename2)
order = ['sqID','hour','wkday','traffic']
data3 = data3[order]
data3['traffic'] = data3['traffic'].fillna(0)

data4 = data3.copy()
data4['traffic'] = df['traffic']


time_step = 3
lentrain = 168*5
epochs = 150
batch_size = 8
ind = np.load('ind.npy')
poolA = list()
tempind = ind.copy()
groupID = [45,46]

#7000
groupID = list(np.arange(21,96))

delete_list = list([27,33,41,49,81,97])

#testlist.remove([1,2])
groupID = [x for x in groupID if x not in delete_list]

print(groupID)

tempcounter = 0
for runcounter in groupID:
  if tempcounter == 0:
    poolA = list(tempind[runcounter,:])
    tempcounter = tempcounter +1
  else:
    poolA = np.concatenate((poolA,tempind[runcounter,:]),axis =0)



print('Cells in the pool:')
print(poolA)
print('Total number:')
print(len(poolA))




PAdf = pd.DataFrame(data=None)
for i in poolA:
    temp = df_1[df_1.sqID==i].traffic
    if len(temp)!=1176:
      #cell_missing.append(i)
      temp = np.lib.pad(temp,(0,1176-len(temp)),'constant',constant_values=(0, 0))
    PAdf['%d'%i] = np.array(temp)

#%%
PAdfs = calDataframeDFS(PAdf)
print('PAdf got')


font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}

fontlabel = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}



tempcounter = 0
for i in np.arange(0,len(groupID)):
  temp_groupweight_loaded = np.load('PAWeight_%d.npy'%groupID[i])
  if i == 0:
    groupweight_loaded = temp_groupweight_loaded
  else:
    groupweight_loaded = np.concatenate((groupweight_loaded,temp_groupweight_loaded),axis =0)

print('Weights loading complete')
print('Number of model weights loaded: %d'%len(groupweight_loaded))

def setweights2(model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, k=3):

    t = []
    
    wtlist1 = convertWeight(model1.get_weights())
    wtlist2 = convertWeight(model2.get_weights())
    wtlist3 = convertWeight(model3.get_weights())
    wtlist4 = convertWeight(model4.get_weights())
    wtlist5 = convertWeight(model5.get_weights())
    wtlist6 = convertWeight(model6.get_weights())
    wtlist7 = convertWeight(model7.get_weights())
    wtlist8 = convertWeight(model8.get_weights())
    wtlist9 = convertWeight(model9.get_weights())
    wtlist10 = convertWeight(model10.get_weights())

    wtlist = (wtlist1 + wtlist2 + wtlist3 + wtlist4 + wtlist5 + wtlist6 + wtlist7 + wtlist8 + wtlist9 + wtlist10)/k
    
    wl1 = wtlist[0:3*20].reshape(3,20)
    wl2 = wtlist[3*20:3*20+5*20].reshape(5,20)
    wl3 = wtlist[3*20+5*20:3*20+5*20+20].reshape(20,-1).flatten()
    wl4 = wtlist[3*20+5*20+20:3*20+5*20+20+5*20].reshape(5,20)
    wl5 = wtlist[3*20+5*20+20+5*20:3*20+5*20+20+5*20+5*20].reshape(5,20)
    wl6 = wtlist[3*20+5*20+20+5*20+5*20:3*20+5*20+20+5*20+5*20+20].reshape(20,-1).flatten()
    wl7 = wtlist[3*20+5*20+20+5*20+5*20+20:3*20+5*20+20+5*20+5*20+20+5*4].reshape(5,4)
    wl8 = wtlist[3*20+5*20+20+5*20+5*20+20+5*4:3*20+5*20+20+5*20+5*20+20+5*4+1*4].reshape(1,4)
    wl9 = wtlist[3*20+5*20+20+5*20+5*20+20+5*4+1*4:].reshape(1,-1).flatten()
    
    
    
    
    t.append(wl1)
    t.append(wl2)
    t.append(wl3)
    t.append(wl4)
    t.append(wl5)
    t.append(wl6)
    t.append(wl7)
    t.append(wl8)
    t.append(wl9)
    
    return t


def setzeroweights():
     
    t = []
    
    #wtlist1 = convertWeight(model1.get_weights())
    #wtlist2 = convertWeight(model2.get_weights())
    #wtlist3 = convertWeight(model3.get_weights())
    #wtlist = (wtlist1 + wtlist2 +wtlist3)/3
    wtlist = np.zeros(428)
    #print('zero weight list:', wtlist)
    #wl1 = wtlist[0:3*40].reshape(3,40)
    #wl2 = wtlist[3*40:3*40+10*40].reshape(10,40)
    #wl3 = wtlist[3*40+10*40:3*40+10*40+40].reshape(40,-1).flatten()
    #wl4 = wtlist[3*40+10*40+40:3*40+10*40+40+10].reshape(10,1)
    #wl5 = wtlist[3*40+10*40+40+10:].reshape(1,-1).flatten()
    
    
    wl1 = wtlist[0:3*20].reshape(3,20)
    wl2 = wtlist[3*20:3*20+5*20].reshape(5,20)
    wl3 = wtlist[3*20+5*20:3*20+5*20+20].reshape(20,-1).flatten()
    wl4 = wtlist[3*20+5*20+20:3*20+5*20+20+5*20].reshape(5,20)
    wl5 = wtlist[3*20+5*20+20+5*20:3*20+5*20+20+5*20+5*20].reshape(5,20)
    wl6 = wtlist[3*20+5*20+20+5*20+5*20:3*20+5*20+20+5*20+5*20+20].reshape(20,-1).flatten()
    wl7 = wtlist[3*20+5*20+20+5*20+5*20+20:3*20+5*20+20+5*20+5*20+20+5*4].reshape(5,4)
    wl8 = wtlist[3*20+5*20+20+5*20+5*20+20+5*4:3*20+5*20+20+5*20+5*20+20+5*4+1*4].reshape(1,4)
    wl9 = wtlist[3*20+5*20+20+5*20+5*20+20+5*4+1*4:].reshape(1,-1).flatten()
    
    
    
    
    t.append(wl1)
    t.append(wl2)
    t.append(wl3)
    t.append(wl4)
    t.append(wl5)
    t.append(wl6)
    t.append(wl7)
    t.append(wl8)
    t.append(wl9)
    
    return t



zeromodel = initModel2(n_features =3, time_step =time_step)
zeromodel_weights = setzeroweights()
zeromodel.set_weights(zeromodel_weights)
#print('zero model weights:', zeromodel.get_weights())



'''
KNN v2: 调整k的数量 section 2【更新模型】——函数定义
'''
def LSTMmt_bl(data3, sid ,initBLweights, time_step, epochs):
    print('Training data length: %d'%lentrain)
    #%Extract 1 Cell's traffic
    ctt = data3[data3.sqID==sid].copy()  
    ctt.drop(['sqID'],axis=1,inplace=True)

    
    ctt.reset_index(drop=True, inplace=True)
    
    #print(ctt)
    min_max_scaler = MinMaxScaler()
    #print('shape:')
    #print(ctt.traffic.shape)
    
    temp_traffic = min_max_scaler.fit_transform(np.array(ctt['traffic']).reshape(-1,1))
    ctt['traffic'] = temp_traffic.reshape(-1)

    ctt_train_X, ctt_test_X, ctt_train_y, ctt_test_y = ctt.values[0:lentrain],ctt.values[840:840+168*2],ctt['traffic'].values[0:lentrain],ctt['traffic'].values[840:840+168*2]
    #ctt_train_X, ctt_test_X, ctt_train_y, ctt_test_y = ctt.values[0:lentrain],ctt.values[lentrain:],ctt['traffic'].values[0:lentrain],ctt['traffic'].values[lentrain:]
    
    ctt_generator_train = TimeseriesGenerator(ctt_train_X, ctt_train_y, length=time_step, batch_size=batch_size, shuffle=False)
    ctt_generator_test = TimeseriesGenerator(ctt_test_X, ctt_test_y, length=time_step, batch_size=batch_size, shuffle=False)

    #Calculate running time
    start1 = time.clock()
    blmodel, wt = LSTMm3_bl(ctt_generator_train, initBLweights, n_features =3, time_step =time_step, epochs = epochs)
    elapsed1 = (time.clock() - start1)
    print("Time used (BL updating):",elapsed1)

    #ctt_y_hat = cttmodel.predict_generator(ctt_generator_test)
    bl_y_hat = blmodel.predict(ctt_generator_test)
    bl_y_true = ctt_generator_test.targets[time_step:]
    RMSE_bl = np.sqrt(mean_squared_error(bl_y_hat,bl_y_true))
    R2_bl = r2_score(bl_y_hat,bl_y_true)
    print('RMSE (function LSTMmt_bl): ',RMSE_bl)
    print('R2 (function LSTMmt_bl): ',R2_bl)
    MAE_bl = mean_absolute_error(bl_y_hat, bl_y_true)
    print('MAE (function LSTMmt_bl): ',MAE_bl)

    return blmodel, RMSE_bl, R2_bl

k=12

os.chdir("/content/gdrive/My Drive/Milan/models_evaluation_v2")


#For Twitter/SMS Guangzhou data, lentrain = 168
#lentrain = 168
#lentrain = 168*5
time_step = 3
batch_size = 16
retrain_epoch = 150

testID = [1684,1884,7121]

print('Test IDs:', testID)

testdf = pd.DataFrame(data=None)

for i in testID:

  temp = df_1[df_1.sqID==i].traffic
  if len(temp)!=1176:
    #cell_missing.append(i)
    temp = np.lib.pad(temp,(0,1176-len(temp)),'constant',constant_values=(0, 0))
  testdf['%d'%i] = np.array(temp)


poolTestdf = testdf

testDFS = calDataframeDFS(poolTestdf)
label_A = np.zeros(len(PAdfs))

#all_id = np.concatenate((poolA,poolB,poolC),axis = 0)
#dataX = np.concatenate((PAdfs,PBdfs,PCdfs),axis = 0)
#dataY = np.concatenate((label_A, label_B, label_C),axis = 0)

all_id = poolA
dataX = PAdfs
dataY = label_A

x_train,y_train=dataX,dataY
x_test,y_test=dataX,dataY





'''
KNN
'''

RMSE_vector = []
MAE_vector = []
R2_vector = []
R22_vector = []
K_list = [12]

K_list = np.arange(1,17)

lentrain_list = [168,168*2,168*3,168*4,168*5]
lentrain_list = [168*5]


for k in K_list:
  knn = neighbors.KNeighborsClassifier(n_neighbors=k)
  knn.fit(x_train, y_train)


  Train_accuracy = knn.score(x_train,y_train)
  Test_accuracy = knn.score(x_test,y_test)

  distances, indices = knn.kneighbors(testDFS)

  #%%
  print('corresponding cell ID:')
  print(np.array(all_id)[indices][0])
  print(np.array(all_id)[indices])
  print('Length:')
  print(len(all_id[indices]))


  id_index = np.zeros(np.max(K_list))
  print('id_index', id_index)
  print('all_id[indices]: ',all_id[indices])
  print('len(all_id[indices]): ',len(all_id[indices]))

  testIDcounter = 0
  MSE = []
  RMSE = []
  MAE = []
  R2 = []
  R22 = []
  for i in all_id[indices]:
    temp_ind_counter = 0
    for j in i:
      id_index[temp_ind_counter] = int(j)
      temp_ind_counter = temp_ind_counter + 1

    print('Final id_index', id_index)

    print('Current testID[testIDcounter]: ', testID[testIDcounter])  
    ct2 = data3[data3.sqID==testID[testIDcounter]].copy()
    
    ct1 = ct2.drop(['sqID'],axis=1)

    ct1.reset_index(drop=True, inplace=True)

    for lentrain in lentrain_list:
        

      train_X, test_X, train_y, test_y = ct1.values[0:lentrain],ct1.values[840:],ct1['traffic'].values[0:lentrain],ct1['traffic'].values[840:]
      print('lentrain: ',lentrain)
      generator_train = TimeseriesGenerator(train_X, train_y, length=time_step, batch_size=batch_size)
      generator_test = TimeseriesGenerator(test_X, test_y, length=time_step, batch_size=batch_size)

      #元学习初始化的model
      ct1model = initModel2(n_features =3, time_step =time_step)

      
      RMSE_bl_ind = []
      MAE_bl_ind = []
      R2_bl_ind = []
      for bl_index in id_index:
        print('Current bl ID: ', bl_index)
        
        if bl_index != 0:
          m_temp = int(bl_index)
          m_temp_path = r'%s.h5' %m_temp
          model_temp = load_model(m_temp_path)
          model_temp_y_hat = model_temp.predict(generator_test)
          RMSE_m_temp = np.sqrt(mean_squared_error(model_temp_y_hat,generator_test.targets[time_step:]))
          R2_m_temp = r2_score(model_temp_y_hat, generator_test.targets[time_step:])
          MAE_m_temp = mean_absolute_error(model_temp_y_hat,generator_test.targets[time_step:])

          RMSE_bl_ind.extend([RMSE_m_temp])
          MAE_bl_ind.extend([MAE_m_temp])
          R2_bl_ind.extend([R2_m_temp])
                             

      print('RMSE_bl_ind: ', RMSE_bl_ind)
      print('MAE_bl_ind: ', MAE_bl_ind)
      print('R2_bl_ind: ', R2_bl_ind)
      print('Min RMSE: ', RMSE_bl_ind[np.argmin(RMSE_bl_ind)])
      print('Corresponding R2: ', R2_bl_ind[np.argmin(RMSE_bl_ind)])
      print('Max R2: ', R2_bl_ind[np.argmax(R2_bl_ind)])
      print('Corresponding meta-cell: ', id_index[np.argmin(RMSE_bl_ind)])


      temp_RMSE = RMSE_bl_ind[np.argmin(RMSE_bl_ind)]
      temp_MAE = MAE_bl_ind[np.argmin(MAE_bl_ind)]
      temp_R2 = R2_bl_ind[np.argmin(RMSE_bl_ind)]
      temp_R22 = R2_bl_ind[np.argmax(R2_bl_ind)]

      RMSE.extend([temp_RMSE])
      MAE.extend([temp_MAE])
      R2.extend([temp_R2])
      R22.extend([temp_R22])

    print('RMSE_%d='%testID[testIDcounter], temp_RMSE)
    print('MAE_%d='%testID[testIDcounter], temp_MAE)
    print('R2_%d='%testID[testIDcounter], temp_R2)

    ct1_y_hat = ct1model.predict_generator(generator_test)

    
    plt.figure()
    plt.tick_params(labelsize=15)
    plt.plot(generator_test.targets[time_step:],'r:.',label = 'Ground truth',linewidth=2.5)
    plt.plot(ct1_y_hat,'cyan',label = '%d'%testID[testIDcounter],linewidth=2.5)
    plt.legend(prop = fontlabel)
    plt.xlabel('Time (hour)',font1)
    plt.ylabel('Traffic',font1)
    elapsed1 = (time.clock() - start1)
    print("Time used (model initialisation):",elapsed1)

    temp_MSE = mean_squared_error(ct1_y_hat,generator_test.targets[time_step:])
    temp_RMSE = np.sqrt(temp_MSE)

    testIDcounter = testIDcounter+1

  print('RMSE:')
  print(RMSE)
  print('MAE:')
  print(MAE)
  print('R2:')
  print(R2)
  print('R22:')
  print(R22)


  print('Convert into array (RMSE vs. len_train) over test cells:')
  print(np.array(RMSE).reshape(len(testID),-1))
  print('Convert into array (MAE vs. len_train) over test cells:')
  print(np.array(MAE).reshape(len(testID),-1))
  print('Convert into array (RMSE vs. len_train) over test cells:')
  print(np.array(R2).reshape(len(testID),-1))
  print('Convert into array (RMSE vs. len_train) over test cells:')
  print(np.array(R22).reshape(len(testID),-1))


  print('Mean RMSE (k=%d):'%k)
  print(np.mean(RMSE))#
  print('Mean MAE (k=%d):'%k)
  print(np.mean(MAE))#
  print('Mean R2 (k=%d):'%k)
  print(np.mean(R2))#
  print('Mean R22 (k=%d):'%k)
  print(np.mean(R22))#

  RMSE_vector.extend([np.mean(RMSE)])
  print('RMSE_vector (K=%d): '%k, RMSE_vector)
  MAE_vector.extend([np.mean(MAE)])
  print('RMSE_vector (K=%d): '%k, MAE_vector)
  R2_vector.extend([np.mean(R2)])
  print('R2_vector (K=%d): '%k, R2_vector)
  R22_vector.extend([np.mean(R22)])
  print('R22_vector (K=%d): '%k, R22_vector)

print('Total number:')
print(len(poolA))

print('RMSE_k= ', RMSE_vector)
print('MAE_k= ', MAE_vector)
print('R2_k= ', R2_vector)
print('R22_k= ', R22_vector)
print('Complete')
