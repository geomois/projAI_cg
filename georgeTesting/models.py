from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.layers import Convolution1D, MaxPooling1D, UpSampling1D
from keras.datasets import mnist
from keras.layers.core import Reshape
import numpy as np

def model1():
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
	autoencoder = Model(input_img, decoded)
	return autoencoder

def model2():
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
	autoencoder = Model(input_img, decoded)
	return autoencoder

def model3():
	#1D convo is different. works. 
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
	return autoencoder

def model4():
	#either pool and upsample in both dim or only in the row
	#using 2D where channels are added one below the other per sec.it works.
	input_img = Input(shape=(1, 2, 28))
	x = Convolution2D(16, 2, 2, activation='relu', border_mode='same')(input_img)#2x28x16
	#x = MaxPooling2D((2, 2), border_mode='same')(x)#1x14x16
	#x = UpSampling2D(size=(2, 2), dim_ordering='th')(x)#2x28x16
	x = MaxPooling2D((2,1),border_mode='valid')(x)#2x14x16
	encoded = Convolution2D(8, 2, 2, activation='relu', border_mode='same')(x)#2x28x8
	encoder = Model(input=input_img,output=encoded)
	#x = MaxPooling2D((2, 2), border_mode='same')(x)#1x14x8
	#x = UpSampling2D(size=(2, 2), dim_ordering='th')(x)#2x28x8
	x = UpSampling2D((2,1), dim_ordering='th')(x)#2x28x8
	x = Convolution2D(16, 2, 2, activation='relu',border_mode='same')(x)#2x28x16
	decoded = Convolution2D(1, 2, 2, activation='sigmoid', border_mode='same')(x)#2x28x1
	autoencoder = Model(input_img, decoded)
	return autoencoder

def applymodel(id,opt,loss,x_train,x_test,y_train,y_test):
	if id==1:
		autoencoder=model1()
	if id==2:
		autoencoder=model2()
	if id==3:
		autoencoder=model3()
	if id==4:
		autoencoder=model4()
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	autoencoder.fit(x_train, x_train,
                	nb_epoch=15,batch_size=128,shuffle=True,
                	validation_data=(x_test, x_test),callbacks=[])
	return autoencoder

def model2use(seclength):
	#having extra channel for images. works.
	input_img = Input(shape=(2, 1, seclength))
	x = Convolution2D(32, 1, 32, activation='relu', border_mode='same')(input_img)#seclengthx32
	x = MaxPooling2D((1,2))(x)#seclength/2x32
	x = Convolution2D(16,1,16,activation='relu',border_mode='same')(x)#seclength/2x16
	x = MaxPooling2D((1,2))(x)#seclength/4x16
	x = Convolution2D(16,1,16,activation='relu',border_mode='same')(x)#seclength/4x16
	x = UpSampling2D((1,2))(x)#seclength/2x16
	x = Convolution2D(32, 1, 32, activation='sigmoid', border_mode='same')(x)#seclength/2x32
	x = UpSampling2D((1,2))(x)#seclengthx32
	decoded = Convolution2D(2,1,16,activation='sigmoid', border_mode='same')(x)#seclengthx2
	autoencoder = Model(input_img, decoded)
	return autoencoder

x = np.random.random((3000,2,1,100))
print "size ",x.shape

def applymodel(autoencoder,x,y,pct,opt,loss,epochs,batch,name):
	x_train=x[0:int(len(x)/7)]
	x_test=x[int(len(x)/7):]
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	autoencoder.fit(x_train, x_train,
        	        nb_epoch=15,batch_size=128,shuffle=True,
                	validation_data=(x_test, x_test),callbacks=[])
	autoencoder.save_weights('keras_w',overwrite=True)

autoencoder=model2use(100)#dimension multiple of 4
applymodel(autoencoder,x,x,0.3,'adadelta','bin',15,128,'keras_w')

trainedautoencoder=model2use(100)
trainedautoencoder.load_weights('keras_w')
 
decoded_imgs = autoencoder.predict(x[10:20])
print "dec ",decoded_imgs.shape

deimgs=trainedautoencoder.predict(x[10:20])
print "deimg",deimgs.shape
