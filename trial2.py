from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.layers import Convolution1D, MaxPooling1D, UpSampling1D
from keras.datasets import mnist
from keras.layers.core import Reshape
import numpy as np

"""
#original for images. works.
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

"""
#having extra channel for images. works.
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
"""

"""
input_img = Input(shape=(1, 1, 28))#tha valo dio kanalia
x = Convolution2D(16, 1, 3, activation='relu', border_mode='same')(input_img)#1x28x16
y = Reshape((28,16))(x)
a = MaxPooling1D(2)(y)#14x16
a = Reshape((16,1,14))
#a = Convolution1D(16,3,activation='relu', border_mode='same')(a)#1x14x16
allos = Model(input_img,a)
#x = Reshape((16,1,28))(y)
x = Convolution2D(16,1,3, activation='relu', border_mode='same')(a)#1x14x16
x = UpSampling2D((1,2))#1x28x16
encoder=Model(input=input_img,output=x)
decoded = Convolution2D(1, 1, 3, activation='sigmoid', border_mode='same')(x)#1x28x1
"""

"""
#using 2D but one channel (channels will be added one below the other per sec).doesnt work.CHECKNOW
input_img = Input(shape=(1, 2, 28))
x = Convolution2D(16, 2, 2, activation='relu', border_mode='same')(input_img)#2x28x16
encoded = Convolution2D(8, 2, 2, activation='relu', border_mode='same')(x)#2x28x8
encoder = Model(input=input_img,output=encoded)
x = Convolution2D(16, 2, 2, activation='relu',border_mode='same')(x)#2x28x16
decoded = Convolution2D(1, 2, 2, activation='sigmoid', border_mode='same')(x)#2x28x1
"""


#1D convo is different. 
input_img=Input(shape=(256,1))
x = Convolution1D(32, 32, activation='relu', border_mode='same')(input_img)#32,256(because step is 256. if i had an input shape 1,256 it would give only 32)
x = MaxPooling1D(2)(x)#32,128
a = Model(input=input_img,output=x)
x = Convolution1D(16, 16, activation='relu', border_mode='same')(x)#16,128
x = MaxPooling1D(2)(x)#16,64
b = Model(input=input_img,output=x)
x = Convolution1D(16, 8, activation='relu', border_mode='same')(x)#16,64
x = MaxPooling1D(2)(x)#16,32
c = Model(input=input_img,output=x)
encoded = Convolution1D(1, 8, activation='relu', border_mode='same')(x)#1,32
encoder = Model(input=input_img,output=encoded)
x = Convolution1D(16, 8, activation='relu', border_mode='same')(encoded)#16,32
x = UpSampling1D(2)(x)#16,64
d = Model(input=input_img,output=x)
x = Convolution1D(16, 16, activation='relu', border_mode='same')(x)#16,64
x = UpSampling1D(2)(x)#16,128
e = Model(input=input_img, output=x)
x = Convolution1D(32, 32, activation='relu', border_mode='same')(x)#32,128
x = UpSampling1D(2)(x)#32,256
decoded = Convolution1D(1, 32, activation='relu',border_mode='same')(x)#1,256

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

"""
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))
"""

#x_train = np.hstack((x_train,x_train))
#x_test = np.hstack((x_test,x_test))

x_train = np.random.random((3000,256,1))
x_test = np.random.random((500,256,1))
print "sizes ",x_train.shape,x_test.shape

y_train = np.random.random((3000,1,8))
y_test = np.random.random((500,1,8))

autoencoder.fit(x_train, x_train,
                nb_epoch=15,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[])

decoded_imgs = autoencoder.predict(x_test)
encoded_imgs = encoder.predict(x_test)
#alla = allos.predict(x_test)
#print "all ", alla.shape
print "dec ",decoded_imgs.shape
print "en ", encoded_imgs.shape
aimgs=a.predict(x_test)
bimgs=b.predict(x_test)
cimgs=c.predict(x_test)
dimgs=d.predict(x_test)
print "a ", aimgs.shape
print "b ", bimgs.shape
print "c ", cimgs.shape
print "d ", dimgs.shape

