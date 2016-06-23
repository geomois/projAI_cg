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
norm_downsampled_data,sample_freq,maxamp=data.downsample()#makes normalization as well

audio_chunks_of1sec = data.split()
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

l=int(x_train.shape[0]*0.3)
x_test=x_train[0:l]
x_train=x_train[l:]

#autoencoder.fit(x_train, x_train, nb_epoch=20,batch_size=128,shuffle=True,validation_data=(x_test, x_test),callbacks=[])
#autoencoder.save_weights('toyCNN.w')
autoencoder.load_weights('toyCNN.w')

pred=autoencoder.predict(x_test)

denorm_x=x_test*maxamp
denorm_pred=pred*maxamp

def tofile(predictions,rate,name):
	signal=np.array([])
	for i in predictions:
		i=np.resize(i,i.shape[0])	
		signal=np.concatenate((signal,i))

	print "signal",signal.shape
	signal=np.asarray(signal, dtype=np.int16)
	wavfile.write(name,sample_freq,signal)

#tofile(denorm_x,sample_freq,'toyforprediction.wav')
#tofile(denorm_pred,sample_freq,'toyprediction.wav')

def plot(before,after):
	from scipy.fftpack import fft, ifft
	plt.figure()
        l=before.shape[0]*before.shape[1]
	f=before.reshape(l)
	plt.subplot(2, 1, 1)
	plt.plot(np.linspace(0,l,l),f)
	plt.subplot(2,1,2)
	f=after.reshape(l)
	plt.plot(np.linspace(0,l,l),f)
	plt.show()

plot(x_test[0:5],pred[0:5])
