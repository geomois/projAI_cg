from __future__ import print_function

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt

from toyPipeline import MyAudio

x_train=MyAudio('toy_data_sines_44_1khz.wav',440,1)
x_train.downsample()#downsample the toy audio
x_train.split(False)#split the toy audio to chunks of 1 sec

x_train=x_train.getInputMatrix()#get chunks as matrix
print("x_train.shape",x_train.shape)


encoding_dim=40

input_img = Input(shape=(880,))
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.activity_l1(10e-5))(input_img)
decoded = Dense(880, activation='tanh')(encoded)

autoencoder = Model(input=input_img, output=decoded)

encoder = Model(input=input_img, output=encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                nb_epoch=20,
                batch_size=50,
                shuffle=True,
                validation_data=(x_train, x_train))

encoded_imgs = encoder.predict(x_train)
decoded_imgs = decoder.predict(encoded_imgs)

print("en ", encoded_imgs.shape)
print("de ", decoded_imgs.shape)

print("in ", x_train.shape)

print(x_train[1])
print(decoded_imgs[1])

#visualize
from scipy.fftpack import fft, ifft
f=fft(x_train[1],100)
plt.subplot(2, 1, 1)
plt.plot(np.linspace(0,100,100),f)
plt.subplot(2,1,2)
f=fft(decoded_imgs[1],100)
plt.plot(np.linspace(0,100,100),f)
plt.show()

