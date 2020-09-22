from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import time
import os 
import csv


# # DRAM Access Class

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
# M = No. of filters 
# N = No. of Channels 
# R = Rows of outout 

# M = No. of filters 
# N = No. of Channels 
# R = Rows of outout 
# C = Rows of input 
def VGG16_GLB(M=np.array([64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 10]), N=np.array([3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512])):
  GLB=32
  D = 1
  B = 2
  M=np.array([64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 10])
  N=np.array([3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512])
  R = np.array([32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2, 1, 1])
  C = np.array([32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2, 1, 1])
  K = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1])
  S = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
  if GLB == 32:
    Tmi = np.array([12,2,13,3,6,2,2,5,1,1,3,3,3,30,10])
    Tni = np.array([3,64,35,55,128,198,198,256,512,512,512,512,512,512,512])
    Tri = np.array([32,7,16,16,8,8,8,4,4,4,2,2,2,1,1])
    Tci = np.array([32,32,16,16,8,8,8,4,4,4,2,2,2,1,1])
    Tmo = np.array([37,32,55,55,161,161,161,512,512,512,512,512,512,512,10])
    Tno = np.array([3,8,3,3,3,3,3,1,1,1,3,3,3,30,503])
    Tro = np.array([32,32,16,16,8,8,8,4,4,4,2,2,2,1,1])
    Tco = np.array([12,11,16,16,8,8,8,4,4,4,2,2,2,1,1])
    Tmw = np.array([64,23,27,14,14,6,6,6,3,3,3,3,3,30,10])
    Tnw = np.array([3,64,64,128,128,256,256,256,512,512,512,512,512,512,512])
    Trw = np.array([7,2,1,1,1,3,3,3,1,1,2,2,2,1,1])
    Tcw = np.array([30,18,9,1,1,3,3,3,4,4,2,2,2,1,1])
  elif GLB == 64:
    Tmi = np.array([28,4,19,3,20,6,6,12,5,5,6,6,6,62,10])
    Tni = np.array([3,64,64,113,128,256,256,256,512,512,512,512,512,512,512])
    Tri = np.array([32,14,16,16,8,8,8,4,4,4,2,2,2,1,1])
    Tci = np.array([32,32,16,16,8,8,8,4,4,4,2,2,2,1,1])
    Tmo = np.array([64,64,113,103,256,256,256,512,512,512,512,512,512,512,10])
    Tno = np.array([1,4,3,5,6,6,6,5,5,5,6,6,6,62,503])
    Tro = np.array([15,14,16,16,8,8,8,4,4,4,2,2,2,1,1])
    Tco = np.array([32,32,16,16,8,8,8,4,4,4,2,2,2,1,1])
    Tmw = np.array([64,48,54,26,26,14,14,14,6,6,6,6,6,62,10])
    Tnw = np.array([3,64,64,128,128,256,256,256,512,512,512,512,512,512,512])
    Trw = np.array([17,3,1,2,3,1,1,1,3,3,2,2,2,1,1])
    Tcw = np.array([27,15,14,9,6,1,1,1,3,3,2,2,2,1,1])
  elif GLB == 128:
    Tmi = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    Tni = np.array([  3.,  64.,  64., 128., 128., 256., 256., 256., 512., 512., 512., 512., 512., 512., 512.])
    Tri = np.array([32., 16., 16., 16.,  8.,  8.,  8.,  4.,  4.,  4.,  2.,  2.,  2.,   1.,  1.])
    Tci = np.array([32., 32., 16., 16.,  8.,  8.,  8.,  4.,  4.,  4.,  2.,  2.,  2.,   1.,  1.])
    Tmo = np.array([ 64.,  64., 128., 128., 256., 256., 256., 512., 512., 512., 512., 512., 512., 512.,  10.])
    Tno = np.array([  1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1., 249.])
    Tro = np.array([16., 16., 16., 16.,  8.,  8.,  8.,  4.,  4.,  4.,  2.,  2.,  2.,    1.,  1.])
    Tco = np.array([32., 32., 16., 16.,  8.,  8.,  8.,  4.,  4.,  4.,  2.,  2.,  2.,    1.,  1.])
    Tmw = np.array([ 64.,  64.,  64.,  43.,  52.,  26.,  26.,  27.,  14.,  14.,  14.,  14.,  14., 103.,  10.])
    Tnw = np.array([  3.,  64.,  64., 128., 128., 256., 256., 256., 512., 512., 512.,  512., 512., 512., 512.])
    Trw = np.array([3., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    Tcw = np.array([15.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., 1.,  1.])
  elif GLB == 256:
    Tmi = np.array([59,16,59,23,47,20,20,26,12,12,13,13,13,126,10])
    Tni = np.array([3,64,64,128,128,256,256,256,512,512,512,512,512,512,512])
    Tri = np.array([32,22,16,16,8,8,8,4,4,4,2,2,2,1,1])
    Tci = np.array([32,32,16,16,8,8,8,4,4,4,2,2,2,1,1])
    Tmo = np.array([64,64,128,128,256,256,256,512,512,512,512,512,512,512,10])
    Tno = np.array([1,16,23,23,20,20,20,12,12,12,13,13,13,126,503])
    Tro = np.array([31,22,16,16,8,8,8,4,4,4,2,2,2,1,1])
    Tco = np.array([32,32,16,16,8,8,8,4,4,4,2,2,2,1,1])
    Tmw = np.array([64,64,64,46,54,28,28,28,14,14,14,14,14,126,10])
    Tnw = np.array([3,64,64,128,128,256,256,256,512,512,512,512,512,512,512])
    Trw = np.array([29,7,14,6,3,1,1,1,1,1,1,1,1,1,1])
    Tcw = np.array([29,32,16,12,6,3,3,3,1,1,1,1,1,1,1])
  elif GLB == 512:
    Tmi = np.array([64,40,128,69,101,48,48,54,26,26,27,27,27,254,10])
    Tni = np.array([3,64,64,128,128,256,256,256,512,512,512,512,512,512,512])
    Tri = np.array([32,32,16,16,8,8,8,4,4,4,2,2,2,1,1])
    Tci = np.array([32,32,16,16,8,8,8,4,4,4,2,2,2,1,1])
    Tmo = np.array([64,64,128,128,256,256,256,512,512,512,512,512,512,512,10])
    Tno = np.array([3,40,64,69,48,48,48,26,26,26,27,27,27,254,503])
    Tro = np.array([32,32,16,16,8,8,8,4,4,4,2,2,2,1,1])
    Tco = np.array([32,32,16,16,8,8,8,4,4,4,2,2,2,1,1])
    Tmw = np.array([64,64,128,83,108,54,54,56,27,27,28,28,28,254,10])
    Tnw = np.array([3,64,64,128,128,256,256,256,512,512,512,512,512,512,512])
    Trw = np.array([31,23,16,12,4,3,3,2,3,3,1,1,1,1,1])
    Tcw = np.array([31,32,16,14,7,7,7,3,4,4,2,2,2,1,1])
  etaI = D*S*S*N*R*C
  etaW = K*K*M*N
  etaO = D*M*R*C
  rui = M*K*K
  ruo = N*K*K
  ruw = D*R*C

  Vil = B*(etaI*np.ceil(M/Tmi)        + etaW*D*np.ceil(R/Tri)*np.ceil(C/Tci)    )
  Vil1= B*(etaO*(2*np.ceil(M/Tmi)-1)  + etaW*D*np.ceil(R/Tri*S)*np.ceil(C/Tci*S))
  Vi  = B*(etaI                       + etaO*(2*np.ceil(N/Tni)-1)               + etaW*D*np.ceil(R/Tri)*np.ceil(C/Tci))
  Vo  = B*(etaI*np.ceil(M/Tmo)        + etaO                                    + etaW*D*np.ceil(R/Tro)*np.ceil(C/Tco))
  Vw  = B*(etaI*np.ceil(M/Tmw)        + etaO*(2*np.ceil(N/Tnw)-1)               + etaW                                )
  NNN = np.shape(M)[0]
  DAi = np.zeros((NNN,))
  DAo = np.zeros((NNN,))
  DAw = np.zeros((NNN,))
  DAi[::2] = Vil[::2]
  DAi[1::2]= Vil1[1::2]
  DAi[NNN-1:] = Vi[NNN-1:]
  DAo[1::2] = Vil[1::2]
  DAo[::2]= Vil1[::2]
  DAo[NNN-1:] = Vi[NNN-1:]

  rs = []
  DM = np.zeros((1,NNN))
  DM1 = np.zeros((1,NNN))
  DS = np.zeros((1,NNN))
  DA = np.zeros((1,NNN))
  range1 = np.arange(0,NNN-1,2)
  range2 = np.arange(NNN-1,NNN)
  for i in range1:
    DM[0,i] = DAi[i]
    DM[0,i+1] = DAi[i+1]
    DM1[0,i+1] = DAo[i+1]
    DM1[0,i+2] = DAo[i+2]
    rs.append('ORO')
    rs.append('IRO')
  for i in range2:
    DM[0,i] = DAi[i]
    DM1[0,i] = DAo[i]
    if rui[i]>ruo[i] and rui[i]>ruw[i]:
      DA[0,i] = DAi[i]
      rs.append('IRO')
    if ruw[i]>=rui[i] and ruw[i]>=ruo[i]:
      DA[0,i] = DAw[i]
      rs.append('WRO')
    if ruo[i]>=rui[i] and ruo[i]>=ruw[i]:
      DA[0,i] = DAo[i]
      rs.append('ORO')
  for i in range(0,NNN):
    if rui[i]>ruo[i] and rui[i]>ruw[i]:
      DS[0,i] = Vi[i]
    if ruw[i]>=rui[i] and ruw[i]>=ruo[i]:
      DS[0,i] = Vo[i]
    if ruo[i]>=rui[i] and ruo[i]>=ruw[i]:
      DS[0,i] = Vw[i]
  for i in range1:
    if (DS[0,i]+DS[0,i+1])<(DM[0,i]+DM[0,i+1]):
      DA[0,i]   = DS[0,i]
      DA[0,i+1] = DS[0,i+1]
    else:
      DA[0,i]   = DM[0,i]
      DA[0,i+1] = DM[0,i+1]
    if (DM[0,i]+DM[0,i+1])>(DM1[0,i]+DM1[0,i+1]) and (DS[0,i]+DS[0,i+1])>(DM1[0,i]+DM1[0,i+1]):
      DA[0,i]   = DM1[0,i]
      DA[0,i+1] = DM1[0,i+1]
  for i in range(0,NNN):
    if (DA[0,i]==0):
      DA[0,i] = DS[0,i]
  return DA/1024/1024

def VGG16_FF(M = np.array([64,64,128,128,256,256,256,512,512,512,512,512,512,512,10]),
N = np.array([3,64,64,128,128,256,256,256,512,512,512,512,512,512,512])
):
  D = 1
  B = 2
  R = np.array([32,32,16,16,8,8,8,4,4,4,2,2,2,1,1])
  C = np.array([32,32,16,16,8,8,8,4,4,4,2,2,2,1,1])
  K = np.array([3,3,3,3,3,3,3,3,3,3,3,3,3,1,1])
  S = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

  Tmi = np.array([12,3,13,11,6,2,6,5,1,1,3,3,3,30,10])
  Tni = np.array([3,64,35,44,128,198,138,256,512,512,512,512,512,512,512])
  Tri = np.array([32,7,16,16,8,8,8,4,4,4,2,2,2,1,1])
  Tci = np.array([32,32,16,16,8,8,8,4,4,4,2,2,2,1,1])
  Tmo = np.array([37,64,32,128,9,9,256,512,512,512,512,512,512,512,10])
  Tno = np.array([3,4,18,5,109,109,5,1,1,3,3,3,3,30,512])
  Tro = np.array([32,22,16,16,8,8,8,4,4,4,2,2,2,1,1])
  Tco = np.array([12,32,14,16,8,8,8,4,4,4,2,2,2,1,1])
  Tmw = np.array([34,64,15,7,10,256,256,6,512,512,512,512,3,512,10])
  Tnw = np.array([3,4,64,128,128,3,6,256,3,3,3,3,512,30,512])
  Trw = np.array([19,22,7,4,5,6,6,3,1,4,2,2,2,1,1])
  Tcw = np.array([22,32,14,16,7,6,6,3,4,4,2,2,2,1,1])

  etaI = D*S*S*N*R*C
  etaW = K*K*M*N
  etaO = D*M*R*C
  rui = M*K*K
  ruo = N*K*K
  ruw = D*R*C

  Vil = B*(etaI*np.ceil(M/Tmi)        + etaW*D*np.ceil(R/Tri)*np.ceil(C/Tci)    )
  Vil1= B*(etaO*(2*np.ceil(M/Tmi)-1)  + etaW*D*np.ceil(R/Tri*S)*np.ceil(C/Tci*S))
  Vi  = B*(etaI                       + etaO*(2*np.ceil(N/Tni)-1)               + etaW*D*np.ceil(R/Tri)*np.ceil(C/Tci))
  Vo  = B*(etaI*np.ceil(M/Tmo)        + etaO                                    + etaW*D*np.ceil(R/Tro)*np.ceil(C/Tco))
  Vw  = B*(etaI*np.ceil(M/Tmw)        + etaO*(2*np.ceil(N/Tnw)-1)               + etaW                                )
  NNN = np.shape(M)[0]
  DAi = np.zeros((NNN,))
  DAo = np.zeros((NNN,))
  DAw = np.zeros((NNN,))
  DAi[::2] = Vil[::2]
  DAi[1::2]= Vil1[1::2]
  DAi[NNN-1:] = Vi[NNN-1:]
  DAo[1::2] = Vil[1::2]
  DAo[::2]= Vil1[::2]
  DAo[NNN-1:] = Vi[NNN-1:]

  rs = []
  DM = np.zeros((1,NNN))
  DM1 = np.zeros((1,NNN))
  DS = np.zeros((1,NNN))
  DA = np.zeros((1,NNN))
  range1 = np.arange(0,NNN-1,2)
  range2 = np.arange(NNN-1,NNN)
  for i in range1:
    DM[0,i] = DAi[i]
    DM[0,i+1] = DAi[i+1]
    DM1[0,i+1] = DAo[i+1]
    DM1[0,i+2] = DAo[i+2]
    rs.append('ORO')
    rs.append('IRO')
  for i in range2:
    DM[0,i] = DAi[i]
    DM1[0,i] = DAo[i]
    if rui[i]>ruo[i] and rui[i]>ruw[i]:
      DA[0,i] = DAi[i]
      rs.append('IRO')
    if ruw[i]>=rui[i] and ruw[i]>=ruo[i]:
      DA[0,i] = DAw[i]
      rs.append('WRO')
    if ruo[i]>=rui[i] and ruo[i]>=ruw[i]:
      DA[0,i] = DAo[i]
      rs.append('ORO')
  for i in range(0,NNN):
    if rui[i]>ruo[i] and rui[i]>ruw[i]:
      DS[0,i] = Vi[i]
    if ruw[i]>=rui[i] and ruw[i]>=ruo[i]:
      DS[0,i] = Vo[i]
    if ruo[i]>=rui[i] and ruo[i]>=ruw[i]:
      DS[0,i] = Vw[i]
  for i in range1:
    if (DS[0,i]+DS[0,i+1])<(DM[0,i]+DM[0,i+1]):
      DA[0,i]   = DS[0,i]
      DA[0,i+1] = DS[0,i+1]
    else:
      DA[0,i]   = DM[0,i]
      DA[0,i+1] = DM[0,i+1]
    if (DM[0,i]+DM[0,i+1])>(DM1[0,i]+DM1[0,i+1]) and (DS[0,i]+DS[0,i+1])>(DM1[0,i]+DM1[0,i+1]):
      DA[0,i]   = DM1[0,i]
      DA[0,i+1] = DM1[0,i+1]
  for i in range(0,NNN):
    if (DA[0,i]==0):
      DA[0,i] = DS[0,i]
  return DA/1024/1024



# # making and decoding of config

# In[8]:


def config_list(): 
    config = [['C',64],['A'],['B'],
              ['C',64],['A'],['B'],['M'],
              ['C',128],['A'],['B'],
              ['C',128],['A'],['B'],['M'],
              ['C',256],['A'],['B'],
              ['C',256],['A'],['B'],
              ['C',256],['A'],['B'],['M'],
              ['C',512],['A'],['B'],
              ['C',512],['A'],['B'],
              ['C',512],['A'],['B'],['M'],
              ['C',512],['A'],['B'],
              ['C',512],['A'],['B'],
              ['C',512],['A'],['B'],['M'],
#               ['D',0.5],
              ['F'],
              ['Dense',512],['A'],['B'],
              ['Dense',10],['L']
             ]
    return config


# In[9]:



def network_config(): 
    config = [['C',64],['A'],['B'],
              ['C',64],['A'],['B'],['M'],
              ['C',128],['A'],['B'],
              ['C',128],['A'],['B'],['M'],
              ['C',256],['A'],['B'],
              ['C',256],['A'],['B'],
              ['C',256],['A'],['B'],['M'],
              ['C',512],['A'],['B'],
              ['C',512],['A'],['B'],
              ['C',512],['A'],['B'],['M'],
              ['C',512],['A'],['B'],
              ['C',512],['A'],['B'],
              ['C',512],['A'],['B'],['M'],
#               ['D',0.5],
              ['F'],
              ['Dense',512],['A'],['B'],
              ['Dense',10],['L']
             ]
    return config


# In[10]:


def update_model(config):
    num_classes = 10
    weight_decay = 0.0005
    x_shape = [32,32,3]
    first_layer = True
    conv_layer=1

    model = Sequential()
    for c in config:
        if (c[0]=='C'):
            layer_name='conv_'+str(conv_layer)
            if first_layer:
                model.add(Conv2D(c[1],(3, 3),
                                 padding='same',input_shape=x_shape,
                                 name=layer_name,
                                 kernel_regularizer=regularizers.l2(weight_decay)))
                first_layer=False 
            else:
                model.add(Conv2D(c[1], (3, 3),
                                 padding='same',
                                 name=layer_name,
                                 kernel_regularizer=regularizers.l2(weight_decay)))
            conv_layer+=1
        elif (c[0]=='A'):
            model.add(Activation('relu'))
        elif (c[0]=='B'):
            model.add(BatchNormalization())
        elif (c[0]=='M'):
            model.add(MaxPooling2D(pool_size=(2, 2)))
        elif (c[0]=='D'):
            model.add(Dropout(c[1]))
        elif (c[0]=='Dense'):
            model.add(Dense(c[1],
                            kernel_regularizer=regularizers.l2(weight_decay)))
        elif (c[0]=='F'):
            model.add(Flatten())
        else:
            model.add(Activation('softmax'))        
    return model   

                
    


# # accuracy of a model 

# In[11]:


def accuracy(test_x, test_y, model):
    result = model.predict(test_x,verbose=1)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(np.equal(predicted_class, true_class)) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)


# # Data Normalization 

# In[12]:


def normalize(X_train,X_test):
    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test
def normalize_production(x):
    mean = 120.707
    std = 64.15
    return (x-mean)/(std+1e-7)


# # Get conv indices from model 

# In[13]:


def get_conv_layer_indices(model):
    conv_layers=[]
    for i,l in enumerate(model.layers):
        if "conv" in l.name: 
            conv_layers.append([l.name,i])
    return conv_layers


# # Get Conv Filters form given model 


# In[15]:


def removing_weights(model,layer_index,rate,score):
    # -- getting weights of the selected layer --
    w=model.layers[layer_index].get_weights()[0]
    b=model.layers[layer_index].get_weights()[1]
    # -- getting weights of the BN layer --
    bn0=model.layers[layer_index+2].get_weights()[0]
    bn1=model.layers[layer_index+2].get_weights()[1]
    bn2=model.layers[layer_index+2].get_weights()[2]
    bn3=model.layers[layer_index+2].get_weights()[3]
    # -- finding prune indices --
    in_channels = w.shape[2]
    out_channels = w.shape[3]
    args=np.argsort(score)
    args=args.reshape(len(args))
    prune_idx=int(rate*out_channels)
    # -- setting unimportant weights to zero --
    w[:,:,:,args[0:prune_idx]]=0
    b[args[0:prune_idx]]=0
    bn0[args[0:prune_idx]]=0
    bn1[args[0:prune_idx]]=0
    bn2[args[0:prune_idx]]=0
    bn3[args[0:prune_idx]]=0
    # -- making new weights --
    w_next=[w,b]
    bn_next=[bn0,bn1,bn2,bn3]
    # -- modify the weights of layers -- 
    model.layers[layer_index].set_weights(w_next)
    model.layers[layer_index+2].set_weights(bn_next)
    return model


# In[16]:


def pruning_layer_wise(conv_indices,scores,prune_ratios):
    model=make_baseline_model()
    for index in range(len(conv_indices)):
        score=scores[index]                   # score for each layer 
        score = score.reshape(len(score))
        layer_index = conv_indices[index][1]   # index of pruned layer 
#         next_layer_index=conv_indices[index+1][1] # index of conv layer next to pruned layer
#         print ("conv index = {} next index = {}"
#               .format(layer_index,next_layer_index))
        prune_ratio = prune_ratios[index]       # pruni
        model=removing_weights(model,layer_index,
                                       prune_ratio,score)
    return model


# # Pruning 

# In[17]:


def saliency_measure_l2_norm(model, conv_indices):
    scores=[]
    for index in conv_indices:
        w=model.layers[index[1]].get_weights()[0]     
        norms=np.linalg.norm(np.linalg.norm(w,2,axis=(0,1)),2,axis=0)
        
        scores.append(norms)
    return scores

def saliency_measure_l1_norm(model, conv_indices):
    scores=[]
    for index in conv_indices:
        w=model.layers[index[1]].get_weights()[0]     
        norms=np.linalg.norm(np.linalg.norm(w,1,axis=(0,1)),1,axis=0)
        
        scores.append(norms)
    return scores


# In[18]:

def pruning_by_fixed_ratio(trained_model,config,x,conv_indices,data,labels,scores):
    acc_conv=[]   # accuracy calculated
    prev_model = update_model(config)  # making a new model
    x=x/100   # converting it into percentage
    for i in range(len(conv_indices)):
        prev_model.set_weights(trained_model.get_weights())
        layer_index=conv_indices[i][1]
#         next_layer_index = conv_indices[i+1][1]
        score=scores[i]
        
        model_new = removing_weights(prev_model,layer_index,
                                    x,score)
        acc_computed=accuracy(model=model_new,test_x=data,
                              test_y=labels)
#         print('For Convolutional layer = {} accuracy = {}'
#               .format(i+1,acc_computed))
        acc_conv.append(acc_computed) 
        
    return acc_conv    
    


# In[19]:


def pruning_for_specific_indices(trained_model,prev_model,ratios,conv_indices,
                                prune_indices,data,labels,scores):
    acc_conv=[]
    for i in range(len(conv_indices)):
        prev_model.set_weights(trained_model.get_weights())
        if i in prune_indices[0]:
            layer_index=conv_indices[i][1]
            score=scores[i]
            r=ratios[i]
            model_new=removing_weights(prev_model,layer_index,r,score)
            acc_computed=accuracy(model=model_new,test_x=data,test_y=labels)
            acc_conv.append(acc_computed)
        else:
            acc_conv.append(-1)
    return acc_conv




# In[20]:


def pruning_by_variable_ratio(trained_model,prev_model,config,ratios,
                              conv_indices,data,labels,scores):
#     print("value of ratios in a function = ",ratios)
    acc_conv = []
#     prev_model = update_model(config)
    for i in range(len(conv_indices)):
        prev_model.set_weights(trained_model.get_weights())
        layer_index=conv_indices[i][1]
        score=scores[i]
        
        r=ratios[i]
        print("finding results for pruning ratio = ",r)
        model_new = removing_weights(prev_model,layer_index,
                                    r,score)
        acc_computed=accuracy(model=model_new,test_x=data,
                              test_y=labels)
        acc_conv.append(acc_computed)
    return acc_conv


# In[21]:


def class_bind_score_calculation(model,conv_indices):
    scores=[]
    i=0
    for index in conv_indices:
        w=model.layers[index[1]].get_weights()[0]
        norms=np.linalg.norm(np.linalg.norm(w,2,axis=(0,1)),2,axis=0)
        a=np.zeros((norms.shape[0],3))
        a[:,0]=int(i)   
        a[:,1]=[int(k) for k in range(norms.shape[0])]
        a[:,2]=norms         
        scores.append(a)
        i+=1
        
    return scores


# In[22]:


def conv_prune(model,layer_index,next_layer_index,rate,score,last):
    w=model.layers[layer_index].get_weights()[0] # weights 
    b=model.layers[layer_index].get_weights()[1] # bias 
    w_bn=model.layers[layer_index+2].get_weights()
    w_next=model.layers[next_layer_index].get_weights()[0] # w_next
    b_next=model.layers[next_layer_index].get_weights()[1] # b_next
    in_channels = w.shape[2] 
    out_channels = w.shape[3]  # number of filters 
    args=np.argsort(score)
    args=args.reshape(len(args))
    prune_idx=int(rate*out_channels)
    new_args=args[prune_idx:,]
    w2=w[:,:,:,new_args]
    b2=b[new_args]
    
    w1_bn=w_bn[0][new_args]
    w2_bn=w_bn[1][new_args]
    w3_bn=w_bn[2][new_args]
    w4_bn=w_bn[3][new_args]
    
    if not last:
        w2_next=w_next[:,:,new_args,:]
        b2_next=b_next
    else:
        w2_next=w_next[new_args,:]
        b2_next=b_next
    return [[w2,b2],[w1_bn,w2_bn,w3_bn,w4_bn],[w2_next,b2_next]]


# In[23]:


def transfer_weights(model_old,model_new,layer_index,next_layer_index, w,w_bn,w_next):
    for i,layer in enumerate(model_new.layers):
#         print ("layer number = ".format(i))
#         print(i)
        if i == layer_index:
            layer.set_weights(w)
        elif i == layer_index+2:
            layer.set_weights(w_bn)
        elif i == next_layer_index:
            layer.set_weights(w_next)
        else: 
            layer.set_weights(model_old.layers[i].get_weights())

    return model_new


# In[24]:


def modify_model_weights(model,layer_index,next_layer_index,rate,score,last):
    weights=model.layers[layer_index].get_weights()
    w=weights[0]
    b=weights[1]
    out_channels = w.shape[3]  # number of filters 
    args=np.argsort(score)
    args=args.reshape(len(args))
    prune_idx=int(rate*out_channels)
    new_args=args[0:prune_idx,]
    w[:,:,:,new_args]=0
    b[new_args]=0
    model.layers[layer_index].set_weights([w,b])
    # --weights of BN layer -------
    weights_bn=model.layers[layer_index+2].get_weights()
    weights_bn[0][new_args]=0
    weights_bn[1][new_args]=0
    weights_bn[2][new_args]=0
    weights_bn[3][new_args]=0
    model.layers[layer_index+2].set_weights(weights_bn)
    if last:
        weights=model.layers[next_layer_index].get_weights()
        w=weights[0]
        b=weights[1]
        w[new_args,:]=0
        b[new_args]=0
        model.layers[next_layer_index].set_weights([w,b])
    else:
        # -- weights of next conv layer ---- 
        weights=model.layers[next_layer_index].get_weights()
        w=weights[0]
        b=weights[1]
        w[:,:,new_args,:]=0
        model.layers[next_layer_index].set_weights([w,b])
    return model


# In[25]:


def senstivity_analysis(model,model_temp,conv_indices,scores,data,labels):
    acc_all_layers=[]
    ratios=[i/10 for i in range(0,11)]
    config_temp=config_list()  # config list 
    for j in range(len(conv_indices)):
#         model_temp=update_model(config=config_temp)
        print ("-- Analysis for layer {} ".format(j+1))
        layer_index=conv_indices[j][1]
        # -- copy weights into temp model --
        model_temp.set_weights(model.get_weights())
        # -- getting layers weights and BN values --
        [w,b]=model.layers[layer_index].get_weights()
        [bn0,bn1,bn2,bn3]=model.layers[layer_index+2].get_weights()
        score=scores[j]
        filters=w.shape[-1]
        args=np.argsort(score)
        args=args.reshape(len(args))
        acc_single_layer=[]
        for r in ratios:
            prune_idx=int(r*filters)
            w[:,:,:,args[0:prune_idx]]=0
            b[args[0:prune_idx]]=0
            bn0[args[0:prune_idx]]=0
            bn1[args[0:prune_idx]]=0
            bn2[args[0:prune_idx]]=0
            bn3[args[0:prune_idx]]=0
            # -- making new weights --
            w_next=[w,b]
            bn_next=[bn0,bn1,bn2,bn3]
            # -- transfering weights to new model --
            model_temp.layers[layer_index].set_weights(w_next)
            model_temp.layers[layer_index+2].set_weights(bn_next)
            acc=accuracy(model=model_temp,test_x=data,test_y=labels)
            acc_single_layer.append(acc)
        acc_all_layers.append(acc_single_layer)
    return acc_all_layers


# In[26]:


def plot_sensitivity_analysis(acc_conv,ratios):
    for i,accuracy in enumerate(acc_conv):
        plt.plot(ratios,accuracy,label='conv_'+str(i+1))
        plt.xlabel("Filters Pruning ratio")
        plt.ylabel("Accuracy (%)")
        plt.title("Pruning Ratio vs Accuracy")
        plt.xlim([0,1.4])
        plt.legend(loc='best')


# In[27]:


def update_conifg(filters):
    config=config_list()
    i=0
    for cfg in config:
        if cfg[0]=='C':
            cfg[1]=filters[i]
            i+=1
    return config


# In[28]:


def get_model_filters(model,conv_indices):
    conv_params_orig=[]  # original filters 
    conv_params_left=[]  # filters left behind after pruning 
    for i in range(len(conv_indices)):
            conv_params_orig.append(model.layers[conv_indices[i][1]].output_shape[3])
            conv_params_left.append(model.layers[conv_indices[i][1]].output_shape[3])
    conv_params_left=np.asarray(conv_params_left)
    conv_params_orig=np.asarray(conv_params_orig)
    return [conv_params_orig,conv_params_left]


# In[29]:


def get_conv_filters_params(filters_list):
    k=3   # filter size = kxk
    c1=3  # first layer channels
    Neurons=51 # Neurons in layer right after last convolution
    total_params_conv = c1*k*k; 
    for j in range (1,len(filters_list)):
        total_params_conv+=k*k*filters_list[j-1]*filters_list[j]
    total_params_conv+=filters_list[-1]*Neurons
    return total_params_conv
        


# # make Baseline Model 

# In[30]:


def make_baseline_model():
    config=config_list()
    model=update_model(config)
    model.load_weights("Baseline.h5")
    return model 


# # Code 



# In[33]:


def get_params_per_filter(conv_params):
    k=3
    params_per_filter=[]
    params_per_filter.append(k*k*3+k*k*conv_params[1])
    for j in range(1,12):
        params_per_filter.append(k*k*conv_params[j-1]+k*k*conv_params[j])
    params_per_filter.append(k*k*conv_params[11]+512*conv_params[12])
    return params_per_filter


# In[34]:


def get_filters_per_layer(conv_indices, model):
    filters_new=[]
    for j in conv_indices:
        filters_new.append(model.layers[j[1]].get_weights()[1].shape[0])
    return filters_new




# In[35]:


def dram_access_models(conv_filters_left,conv_filters_new, dram_obj):
    dram_access_layers = []
    R = np.array([32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 1, 1, 1,1,1])
    C = np.array([32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 1, 1, 1,1,1])
    for j,kernels in enumerate(conv_filters_new):
        M = conv_filters_left.copy()
        M=M.tolist()
        M.extend(list([512,10]))
        N = [3]
        N.extend(M[0:-1])
        M[j]=kernels
        N[j+1]=kernels
        dram_access_layers.append(sum(dram_obj.DRAM_Access_WRO(M=M,C=C,N=N,R=R)[0]))
    return dram_access_layers


# In[36]:


def dram_access_model(conv_filters,dram_obj):
    R = np.array([32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2,1,1])
    C = np.array([32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2,1,1])
    M = conv_filters_left.copy()
    M=M.tolist()
    M.extend(list([512,10]))
    N = [3]
    N.extend(M[0:-1])
    V=dram_obj.DRAM_Access_WRO(M=M,C=C,N=N,R=R)[0]
    return V 
    

# In[90]:


def prune_one_layer(model_prev,model,Acc_b,delta_A,layer_index,score,x_list,data,labels): 
    acc_conv=[]
    for x in x_list:
        model_prev.set_weights(model.get_weights())
        model_new = removing_weights(model_prev,layer_index,x,score)
        acc_computed=accuracy(model=model_new,test_x=data,test_y=labels)
        if (abs(acc_computed-Acc_b)>=delta_A):
            return [acc_conv,x]
        acc_conv.append(acc_computed)
    return [acc_conv,x] 

def dram_based_pruning(model,model_prev,scores,v,data,labels,conv_indices,prev_acc,delta_A,ratios_matrix): 
    indices = np.argsort(v[0:13])
    L = len(indices)
    for i in range(L):
        idx = indices[L-1-i]
        print ('---- For layer = {} ----'.format(idx))
        score = scores[idx]
        layer_index = conv_indices[idx][1]
        x_list = ratios_matrix[:,idx]
        [acc_conv,r]=prune_one_layer(model_prev,model,prev_acc,delta_A,layer_index,score,x_list,data,labels)
        if len(acc_conv) > 0:
            return [acc_conv,r,idx]
        else:
            print ('changing the layer ')
            if i== L-1: 
                return [acc_conv,0,L-1-i] 
    
def make_pruned_model(model_prev,conv_indices):
    new_scores=saliency_measure_l2_norm(model_prev,conv_indices)
    
    not_zero_indices=[]
    filters_retain = []
    for new_score in new_scores:
        not_zero_indices.append(np.where(new_score!=0)[0])
        filters_retain.append(len(np.where(new_score!=0)[0]))
    config_new = config_list()
    for i,index in enumerate(conv_indices):
        config_new[index[1]][1]=(filters_retain[i])
    print(config_new)
    model_new=update_model(config_new)
    model_temp = update_model(config_new)
    # filtes_removed_each_step[-1]=filters_left
    # --------------------------------------------------------------
    # ----- Transfering weights to new model -----
    [w,b]=model_prev.layers[conv_indices[0][1]].get_weights()
    w=w[:,:,:,not_zero_indices[0]]
    b=b[not_zero_indices[0]]
    model_new.layers[conv_indices[0][1]].set_weights([w,b])
    bn_0=model_prev.layers[conv_indices[0][1]+2].get_weights()[0][not_zero_indices[0]]
    bn_1=model_prev.layers[conv_indices[0][1]+2].get_weights()[1][not_zero_indices[0]]
    bn_2=model_prev.layers[conv_indices[0][1]+2].get_weights()[2][not_zero_indices[0]]
    bn_3=model_prev.layers[conv_indices[0][1]+2].get_weights()[3][not_zero_indices[0]]
    model_new.layers[conv_indices[0][1]+2].set_weights([bn_0,bn_1,bn_2,bn_3])
    for i in range(1,len(conv_indices)):
        [w,b]=model_prev.layers[conv_indices[i][1]].get_weights()
        w=w[:,:,:,not_zero_indices[i]]
        b=b[not_zero_indices[i]]
        w=w[:,:,not_zero_indices[i-1],:]
        model_new.layers[conv_indices[i][1]].set_weights([w,b])
        bn_0=model_prev.layers[conv_indices[i][1]+2].get_weights()[0][not_zero_indices[i]]
        bn_1=model_prev.layers[conv_indices[i][1]+2].get_weights()[1][not_zero_indices[i]]
        bn_2=model_prev.layers[conv_indices[i][1]+2].get_weights()[2][not_zero_indices[i]]
        bn_3=model_prev.layers[conv_indices[i][1]+2].get_weights()[3][not_zero_indices[i]]
        model_new.layers[conv_indices[i][1]+2].set_weights([bn_0,bn_1,bn_2,bn_3])

    [w,b]=model_prev.layers[conv_indices[-1][1]+5].get_weights()
    w=w[not_zero_indices[12],:]
    # b=b[not_zero_indices[12]]
    model_new.layers[conv_indices[-1][1]+5].set_weights([w,b])
    for j in range(conv_indices[-1][1]+6,len(model_prev.layers)):
        model_new.layers[j].set_weights(model_prev.layers[j].get_weights())
    return [model_new,model_temp] 



# In[97]:


def get_pruning_ratio(x_list,conv_filters_left,conv_orig_matrix): 
    new_filters = []
    for x in x_list: 
        new_filters.append((1-x)*conv_filters_left)
    new_filters=np.asarray(new_filters)
    ratios_matrix=1-np.divide(new_filters,conv_orig_matrix)
    return ratios_matrix
