from keras.layers import Input, Convolution1D, UpSampling1D, AveragePooling1D, MaxPooling1D
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
import numpy as np
from keras import callbacks
from keras.layers.advanced_activations import LeakyReLU, PReLU

class kerasModel:
    def __init__(self, waves, annotations, validation, vAnnot):
        self.signals=waves[:,0:13208]#in order to do 3 poolings of length 2
        self.annotations=annotations[:,0:13208]
        self.validation=validation[:,0:13208]
        self.validAnnotation=vAnnot[:,0:13208]
        self.autoencoder=None
	print "kkk",len(self.signals[1])

    def buildAutoEncoder(self,init_filter=32,filter_size=2,loss_f='binary_crossentropy',optim_f='adadelta',f='relu'):
        input_au=Input(shape=(self.signals.shape[1], 1))

	if filter_size==64:	
	        x = Convolution1D(init_filter/8, filter_size, activation=f, border_mode='same')(input_au)#filter/4*shape
	        x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)#filter/4*shape/2
 
        x = Convolution1D(init_filter/4, filter_size, activation=f, border_mode='same')(input_au)#filter/4*shape
        x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)#filter/4*shape/2
        x = Convolution1D(init_filter/2, filter_size, activation=f, border_mode='same')(x)#filter/2*shape/2
	x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)#filter/2*shape/4
        x = Convolution1D(1, filter_size, activation=f, border_mode='same')(x)#1*shape/4

        x = Convolution1D(init_filter/2, filter_size, activation=f, border_mode='same')(x)#filter/2*shape/4
        x = UpSampling1D(length=2)(x)#filter/2*shape/2
        x = Convolution1D(init_filter/4, filter_size, activation=f, border_mode='same')(x)#filter/4*shape/2
	x = UpSampling1D(length=2)(x)#filter/4*shape

	if filter_size==64:
	       x = Convolution1D(init_filter/8, filter_size, activation=f, border_mode='same')(x)#filter/4*shape/2
 	       x = UpSampling1D(length=2)(x)#filter/4*shape

        x = Convolution1D(1, filter_size, activation=f,border_mode='same')(x)#1*shape

        self.autoencoder = Model(input_au, x)
        self.autoencoder.compile(optimizer=optim_f, loss=loss_f)

    def autoEncoderTrain(self,target=None,epochs=20,batch=128,name='ae_weights.w'):
    	if target is None:
            target=self.signals
        print 'Training..'
	target=target[:,0:13208]
	#history=callbacks.History()
	earlystop=callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        #checkpoint=callbacks.ModelCheckpoint(name, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
	print("in train",self.signals.shape,target.shape)
	self.autoencoder.fit(self.signals, target, nb_epoch=epochs, batch_size=batch, shuffle=True, callbacks=[earlystop], validation_data=(self.validation, self.validAnnotation))	
        self.autoencoder.save_weights(name, True)
	
    def loadWeights(self,name):
	self.autoencoder.load_weights(name)
    
    def evaluate(self):
	print "accuracy ",self.autoencoder.evaluate(self.validation, self.validAnnotation)
	pred=self.autoencoder.predict(self.validation)
	print pred.shape
	print pred[0:2]
	pred[pred>=0.5]=1
	pred[pred<0.5]=0
	print pred[0:2]
	diff=self.validAnnotation-pred	
	print diff[0:2]
	acc=np.sum(np.abs(diff))
	print "accuracy ",acc/(diff.shape[0]*diff.shape[1]) 
