from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
import numpy as np
"""
input_img = Input(shape=(1, 28, 28))
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)#1x28x28x16
x = MaxPooling2D((2, 2), border_mode='same')(x)#1x14x14x16
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)#1x14x14x8
x = MaxPooling2D((2, 2), border_mode='same')(x)#1x7x7x8
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)#1x7x7x8
encoded = MaxPooling2D((2, 2), border_mode='same')(x)#1x4x4x8
encoder=Model(input=input_img,output=encoded)#nomizo
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)#1x4x4x8
x = UpSampling2D((2, 2))(x)#1x8x8x8
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)#1x8x8x8
x = UpSampling2D((2, 2))(x)#1x16x16x8
x = Convolution2D(16, 3, 3, activation='relu')(x)#1x16x16x16
x = UpSampling2D((2, 2))(x)#1x32x32x16
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)#1x32x32x1
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
"""


input_img = Input(shape=(2, 28, 28))
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)#28x28x16
x = MaxPooling2D((2, 2), border_mode='same')(x)#14x14x16
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)#14x14x8
x = MaxPooling2D((2, 2), border_mode='same')(x)#7x7x8
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)#7x7x8
encoded = MaxPooling2D((2, 2), border_mode='same')(x)#4x4x8
encoder=Model(input=input_img,output=encoded)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)#4x4x8
x = UpSampling2D((2, 2))(x)#8x8x8
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)#8x8x8
x = UpSampling2D((2, 2))(x)#16x16x8
x = Convolution2D(16, 3, 3, activation='relu')(x)#16x16x16
x = UpSampling2D((2, 2))(x)#32x32x16
decoded = Convolution2D(2, 3, 3, activation='sigmoid', border_mode='same')(x)#32x32x2
#decoded = Reshape((2,))?????


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))
x_train = np.hstack((x_train,x_train))
x_test = np.hstack((x_test,x_test))
print "sizes ",x_train.shape,x_test.shape

autoencoder.fit(x_train, x_train,
                nb_epoch=15,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[])

decoded_imgs = autoencoder.predict(x_test)
encoded_imgs = encoder.predict(x_test)#nomizo

print "ok "
print "dec ",decoded_imgs.shape
print "en ", encoded_imgs.shape

#to take output of a hidden layer
#model2= Convolution2D(16, 3, 3, activation='relu', border_mode='same', weights=autoencoder.layers[0].get_weights())(input_img)
#X_batch=x_test[0:10,:,:,:]
#activations = model2._predict(X_batch)

#print activations.shape
