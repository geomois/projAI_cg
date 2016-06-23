from keras.layers import Convolution1D, UpSampling1D, AveragePooling1D, MaxPooling1D
from keras.models import Sequential
from keras.layers import Input,Dense
from keras.models import Model
from keras import regularizers
from keras.optimizers import SGD
from scipy.io import wavfile
import numpy as np
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt

from toyPipeline import MyAudio

def plot(before,after):
        plt.figure()
        l=before.shape[0]*before.shape[1]
        f=before.reshape(l)
	print "f: ",f
        plt.subplot(2, 1, 1)
        plt.plot(np.linspace(0,l,l),f)
        plt.subplot(2,1,2)
        f=after.reshape(l)
        print "f: ",f
        plt.plot(np.linspace(0,l,l),f)
        plt.show()

def tofile(predictions,rate,name):
        signal=np.array([])
        for i in predictions:
                i=np.resize(i,i.shape[0])
                signal=np.concatenate((signal,i))
	signal=np.asarray(signal)
        print "signal ",signal	
	print "shape ",signal.shape
        wavfile.write(name,rate,signal)


sr,au=wavfile.read('toy_data_sines_44_1khz.wav')
data=MyAudio(sr,au,1)
norm_downsampled_data,sample_freq,maxamp=data.downsample()#makes normalization as well

data.split()
data_input=data.getInputMatrix()

input_img = Input(shape=(880,1))

x = Convolution1D(16, 32, activation='tanh', border_mode='same')(input_img)#16x880
x = MaxPooling1D(2, border_mode='valid')(x) #outvol=440x16
x = Convolution1D(8,16, activation='tanh', border_mode='same')(x)#8x440
x = MaxPooling1D(2, border_mode='valid')(x) #outvol=220x8
x = Convolution1D(4,16, activation='tanh', border_mode='same')(x)#4x220
encoded = MaxPooling1D(2, border_mode='valid')(x) #outvol=110x4
x = Convolution1D(4,16, activation='tanh', border_mode='same')(encoded)#4x110
x = UpSampling1D(2)(x) #outvol=220x4
x = Convolution1D(8,16, activation='tanh', border_mode='same')(x)#8x220
x = UpSampling1D(2)(x) #outvol=440x8
x = Convolution1D(16,32, activation='tanh',border_mode='same')(x)#16x440
x = UpSampling1D(2)(x) #outvol=880x16
decoded = Convolution1D(1,16, activation='tanh', border_mode='same')(x)#1x880

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='rmsprop', loss='mse')

x_train = np.reshape(data_input, (data_input.shape[0], data_input.shape[1],1))

split=int(x_train.shape[0]*0.3)
x_test=x_train[0:split]
x_train=x_train[split:]

#autoencoder.fit(x_train, x_train, nb_epoch=20,batch_size=128,shuffle=True,validation_data=(x_test, x_test),callbacks=[])
#autoencoder.save_weights('toyCNN.w',overwrite=True)
autoencoder.load_weights('toyCNN.w')

predictions=autoencoder.predict(x_test)

denorm_x=x_test*maxamp
denorm_predictions=predictions*maxamp

tofile(denorm_x,880,'toyforprediction.wav')
tofile(denorm_predictions,880,'toyprediction.wav')

plot(x_test[0:20],predictions[0:20])

