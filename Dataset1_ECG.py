import numpy as np 
import pandas as pd 
import os

import matplotlib.pyplot as plt
import csv
import itertools
import collections

import pywt
from scipy import stats

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Conv1D, AvgPool1D, Flatten, Dense, Dropout, Softmax
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras import regularizers


%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import LeakyReLU
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Add, Embedding, Conv1DTranspose, RepeatVector, Softmax, Conv1D, \
    Flatten, UpSampling1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import BatchNormalization

from keras.models import model_from_json
from sklearn.metrics import accuracy_score
import warnings
import glob
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')


def mean_column( X, col_num ):
	mean=0.0
	c=0
	for i in range(0,452):
		if(X[i][col_num] !="?"):	
			mean=mean+X[i][col_num].astype(float)
			c=c+1
	mean=mean/c
	return mean

def standard_deviation_column( X, col_num ,mean):
	sd=0.0
	c=0
	for i in range(0,452):
		if(X[i][col_num] !="?"):	
			sd=(X[i][col_num].astype(float)-mean)**2
			c=c+1
	sd=sd/(c-1)
	sd=sd**0.5
	return sd
	
	
def convert_strarr_floatarr( arr, X):
	for i in range(0,452):
		for j in range(0,278):
			if(arr[i][j]=="?"):	
				X[i][j]=0.0
			else:
				X[i][j]=arr[i][j].astype(float)
	return
					
			
reader=csv.reader(open("/content/drive/MyDrive/arrhythmia.csv","r"),delimiter=",")
arr=list(reader)
arr=np.array(arr)
data=np.zeros((452,2))
c=0
for i in range(0,452):
	for j in range(0,279):
		if(arr[i][j] =="?"):	
			data[c][0]=i
			data[c][1]=j
			c=c+1

#majority of the values are missing so delete coulmn 13
#find the columns with missing values			
for i in range(0,c):
	if(data[i][1]!=13):
		print(data[i][0],data[i][1])

#remove coulmn 13
arr = np.delete(arr,13,1)

#create feature matrix
X=np.zeros((452,278),dtype=float)
convert_strarr_floatarr(arr,X)

#create result vector
y=np.zeros((452),dtype=int)
for i in range(0,452):
	y[i]=arr[i][278].astype(int)
print (y)

#find the columns with missing values			
for i in range(0,c):
	if(data[i][1]!=13):
		print(data[i][0],data[i][1])
			
#calculate mean for column 13(initially 14),11,10,12
mean=mean_column(X,13)
print ("mean="+str(mean))
sd=standard_deviation_column(X,13,mean)

for i in range(0,452):
	if(arr[i][13]=="?"):
		val = np.random.normal(mean,sd,1)
		print (val)
		X[i][13]=(val).astype(int)
		print (X[i][13])
		
mean=mean_column(X,10)
print ("mean="+str(mean))
sd=standard_deviation_column(X,10,mean)

for i in range(0,452):
	if(arr[i][10]=="?"):
		val = np.random.normal(mean,sd,1)
		print(val)
		X[i][10]=(val).astype(int)
		print (X[i][10])

mean=mean_column(X,11)
print ("mean="+str(mean))
sd=standard_deviation_column(X,11,mean)

for i in range(0,452):
	if(arr[i][11]=="?"):
		val = np.random.normal(mean,sd,1)
		print (val)
		X[i][11]=(val).astype(int)
		print (X[i][11])

mean=mean_column(X,12)
print ("mean="+str(mean))
sd=standard_deviation_column(X,12,mean)

for i in range(0,452):
	if(arr[i][12]=="?"):
		val = np.random.normal(mean,sd,1)
		print (val)
		X[i][12]=(val).astype(int)
		print (X[i][12])
#reduce number of classes
for i in range(0,452):
	if (y[i]>=14):
		y[i]=y[i]-3

np.savetxt("feature.csv", X, fmt='%s', delimiter=",")
np.savetxt("target_output.csv", y, fmt='%s', delimiter=",")
#result=numpy.array(x).astype("str")



import csv
import numpy as np

import pandas
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

def convert_strarr_floatarr( arr, X):
	for i in range(0,452):
		for j in range(0,278):
				X[i][j]=arr[i][j].astype(float)
	return

#Feature extraction


#create feature matrix
reader=csv.reader(open("feature.csv","r"),delimiter=",")
X=list(reader)
X=np.array(X)
X=X.astype(float)

#create result vector
reader=csv.reader(open("target_output.csv","r"),delimiter=",")
Y=list(reader)
Y=np.array(Y)
Y=Y.astype(int)

	

#applying PCA to get pricipal attributes
pca = PCA(n_components=50)
X=pca.fit_transform(X)

print (pca.explained_variance_ratio_)

np.savetxt("reduced_features.csv",X, fmt='%s', delimiter=",")


readerF=csv.reader(open("reduced_features.csv","r"),delimiter=",")
X=list(readerF)
X=np.array(X)
X=X.astype(float)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

readerL=csv.reader(open("target_output.csv","r"),delimiter=",")
Y=list(readerL)
Y=np.array(Y)
Y=Y.astype(int)

#splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.40, random_state=0)


Y_train=np.transpose(Y_train)
Y_test=np.transpose(Y_test)

#one hot encoding for multiclass (Training labels)
one_hot_encoded=list()
for value in range (0,Y_train.shape[1]):
	out=list()
	out=[0 for i in range(13)]
	out[Y_train[0][value]-1]=1
	one_hot_encoded.append(out)
 

Y_train=one_hot_encoded
Y_train=np.array(Y_train)



#one hot encoding for multiclass (Testing labels)
one_hot_encoded2=list()
for value in range (0,Y_test.shape[1]):
	out2=list()
	out2=[0 for i in range(13)]
	out2[Y_test[0][value]-1]=1
	one_hot_encoded2.append(out2)
 
Y_test=one_hot_encoded2
Y_test=np.array(Y_test)

print("shape of XTrain ="+str(X_train.shape))
print("shape of YTrain ="+str(Y_train.shape))

print("shape of XTest ="+str(X_test.shape))
print("shape of YTest ="+str(Y_test.shape))

# X_train=np.transpose(X_train)
# Y_train=np.transpose(Y_train)

# X_test=np.transpose(X_test)
# Y_test=np.transpose(Y_test)


# print("T shape of XTrain ="+str(X_train.shape))
# print("T shape of YTrain ="+str(Y_train.shape))

# print("T shape of XTest ="+str(X_test.shape))
# print("T shape of YTest ="+str(Y_test.shape))


###############LSTM

# ac = tf.keras.layers.LeakyReLU(alpha=0.9)
model1 = tf.keras.Sequential()
model1.add(tf.keras.layers.LSTM(units = 64, activation='relu', input_shape = (50,1)))
# model1.add(tf.keras.layers.Dropout(.5, noise_shape=None, seed=None))
model1.add(Dense(13, activation='relu'))
model1.add(tf.keras.layers.BatchNormalization())
model1.add(Dense(13, activation='softmax'))
model1.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model1.compile(optimizer=opt,
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# history = model1.fit(train_x, train_y, epochs = 250, validation_split = 0.40, batch_size = 25, verbose = 1)


##########CNN


model3 = tf.keras.Sequential()
model3.add(tf.keras.Input(shape = (50,1)))

model3.add(Conv1D(2, kernel_size=2, strides=1, padding='same', activation='relu'))
model3.add(MaxPooling1D(pool_size=1, strides=1))

model3.add(Conv1D(2, kernel_size=4, strides=1, padding='same', activation='relu'))
model3.add(MaxPooling1D(pool_size=1, strides=1))

model3.add(Conv1D(2, kernel_size=8, strides=1, padding='same', activation='relu'))
model3.add(MaxPooling1D(pool_size=1, strides=1))

model3.add(tf.keras.layers.Dropout(.3, noise_shape=None, seed=None))
model3.add(tf.keras.layers.Flatten(data_format=None))
model3.add(tf.keras.layers.BatchNormalization())
model3.add(Dense(13, activation='relu'))
model3.add(Dense(13, activation='softmax'))
model3.summary()

model3.compile(optimizer=opt,
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# history2 = model3.fit(train_x, train_y, epochs = 250, validation_split = 0.40, batch_size = 25, verbose = 1)


############ Merging all models

mergedOut = Add()([model1.output, model3.output])
newModel = Model([model1.input, model3.input], mergedOut)
newModel.summary()

newModel.compile(optimizer=opt,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

newModel.fit([X_train, X_train], Y_train, epochs=500, batch_size=20, shuffle=False)

preddd = newModel.predict([X_test, X_test])
print(preddd.shape)
preddd = np.argmax(preddd, axis=1)
print(preddd)


# #   Final Accuracy


max_testNM = np.argmax(Y_test, axis=1)
print(max_testNM)

Accuracy = accuracy_score(max_testNM, preddd)
NAccuracy = Accuracy * 100
print('Merged Model Accuracy:', NAccuracy, '%')
