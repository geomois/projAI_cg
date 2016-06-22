from keras.layers import ZeroPadding1D,Input, Convolution1D, UpSampling1D, AveragePooling1D, MaxPooling1D
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
import numpy as np
import pdb
import theano


class kModel:
    def __init__(self, waves, rate, annotations, validation, vAnnot, vRate):
        self.data = []
        self.signals=waves
        self.rate=rate
        self.annotations=annotations
        self.validation=validation
        self.validAnnotation=vAnnot
        self.validRate=vRate
        self.autoencoder=None

    # noinspection PyPep8Naming
    def buildAutoEncoder(self,train,inputShape=None,target=None):
        if inputShape is None:
            input_au=Input(shape=(self.signals.shape[1], 1))
        else:
            input_au=inputShape
        self.outLayers={}
        input_au=Input(shape=(self.signals.shape[1], 1))
        encoded1 = Convolution1D(32, 2, activation='relu', border_mode='same',name='conv1')(input_au)#16
        self.outLayers['encoder1']=Model(input=input_au,output=encoded1)
        x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(encoded1)
        encoded2 = Convolution1D(32, 2, activation='relu', border_mode='same',name='conv2')(x)#16
        self.outLayers['encoder2'] = Model(input=input_au, output=encoded2)
        encoded3 = Convolution1D(8, 2, activation='relu', border_mode='same',name='conv3')(encoded2)#8
        self.outLayers['encoder3']= Model(input=input_au, output=encoded3)

        encoded4 = Convolution1D(8, 2, activation='relu', border_mode='same',name='conv4')(encoded3)#8
        self.outLayers['encoder4']= Model(input=input_au,output=encoded4)
        x = UpSampling1D(length=2)(encoded4)
        encoded5 = Convolution1D(16, 2, activation='relu', border_mode='same',name='conv5')(x)#16
        self.outLayers['encoder5']=Model(input=input_au,output=encoded5)
        decoded = Convolution1D(32, 2, activation='relu',border_mode='same')(encoded5)#28
        self.autoencoder = Model(input_au, decoded)
        self.autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

        if target is None:
            target=self.signals
        if train:
            print 'Training..'
            self.autoencoder.fit(self.signals, target, nb_epoch=15, batch_size=64, shuffle=True, callbacks=[])
            self.autoencoder.save_weights("kModel_weights.w", True)
        else:
            self.autoencoder.load_weights('kModel_weights.w')
        print 'done'
    def predict(self):
        self.buildAutoEncoder(False)
        predictions = self.autoencoder.predict_on_batch(self.validation)
        error = mean_squared_error(np.resize(self.validAnnotation, (len(self.validAnnotation), self.validRate)), np.resize
            (predictions, (len(predictions), self.validRate)))
        print 'error :', error
    
    def getModel(self):
        return self.autoencoder,self.outLayers

