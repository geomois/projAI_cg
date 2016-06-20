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
        self.signals=waves
        self.annotations=annotations
        self.validation=validation
        self.validAnnotation=vAnnot
        self.autoencoder=None

    def buildAutoEncoder(self,init_filter=32,filter_size=2,loss_f='mean_squared_error',optim_f='adadelta',f='relu'):
        input_au=Input(shape=(self.signals.shape[1], 1))

        x = Convolution1D(init_filter/4, filter_size, activation=f, border_mode='same')(input_au)#16
        x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
        x = Convolution1D(init_filter/2, filter_size, activation=f, border_mode='same')(x)#16
#	x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
#        x = Convolution1D(init_filter, filter_size, activation=f, border_mode='same')(x)#8

#        x = Convolution1D(init_filter, filter_size, activation=f, border_mode='same')(x)#8
#        x = UpSampling1D(length=2)(x)
        x = Convolution1D(init_filter/2, filter_size, activation=f, border_mode='same')(x)#16
	x = UpSampling1D(length=2)(x)
	x = Convolution1D(init_filter, filter_size, activation=f, border_mode='same')(x)#16
        x = Convolution1D(1, filter_size, activation=f,border_mode='same')(x)#28

        self.autoencoder = Model(input_au, x)
        self.autoencoder.compile(optimizer=optim_f, loss=loss_f)

    def autoEncoderTrain(self,target=None,epochs=20,batch=128,name='ae_weights.w'):
    	if target is None:
            target=self.signals
        print 'Training..'
	#history=callbacks.History()
	#earlystop=callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        #checkpoint=callbacks.ModelCheckpoint(name, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
	self.autoencoder.fit(self.signals, target, nb_epoch=epochs, batch_size=batch, shuffle=True, callbacks=[],validation_data=(self.validation[0:100], self.validAnnotation[0:100]))	
        #self.autoencoder.save_weights(name, True)
	
    def loadWeights(self,name):
	self.autoencoder.load_weights(name)
    
