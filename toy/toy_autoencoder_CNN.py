
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
sr,au=wavfile.read('toy_data_sines_44_1khz.wav')
data=MyAudio(sr,au,1)
downsampled_data,sample_freq=data.downsample()
audio_chunks_of1sec = data.split()
data_input=data.getInputMatrix()
print "NOW:",data_input.shape


input_img = Input(shape=(880,1))

x = Convolution1D(16, 40, activation='relu', border_mode='same')(input_img)#16x880
x = MaxPooling1D(2, border_mode='valid')(x) #outvol=440x16
x = Convolution1D(8,20, activation='relu', border_mode='same')(x)
x = MaxPooling1D(2, border_mode='valid')(x) #outvol=220x8
x = Convolution1D(4,10, activation='relu', border_mode='same')(x)
encoded = MaxPooling1D(2, border_mode='valid')(x) #outvol=110x4
x = Convolution1D(4,10, activation='relu', border_mode='same')(encoded)
x = UpSampling1D(2)(x) #outvol=220x4
x = Convolution1D(8,20, activation='relu', border_mode='same')(x)
x = UpSampling1D(2)(x) #outvol=440x8
x = Convolution1D(16,40, activation='relu')(x)
x = UpSampling1D(2)(x) #outvol=880x16
decoded = Convolution1D(1,4, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='rmsprop', loss='mse')

x_train = np.reshape(data_input, (data_input.shape[0], data_input.shape[1],1))
print "train", x_train.shape

autoencoder.fit(x_train, x_train,
                nb_epoch=20,
                batch_size=64,
                shuffle=True,
                validation_data=(x_train, x_train),
                callbacks=[])

print "ok"
