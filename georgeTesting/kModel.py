from keras.layers import Input, Convolution1D, UpSampling1D, AveragePooling1D, MaxPooling1D
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
import numpy as np

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
    def buildAutoEncoder(self,train,ten,inputShape=None,target=None):
        if inputShape is None:
            input_au=Input(shape=(self.signals.shape[1], 1))
        else:
            input_au=inputShape
        print ten.shape
        firstLayer=Convolution1D(64, 64, activation='relu', border_mode='same',name='conv1')
        firstLayer.set_input(ten,shape=input_au)
        self.autoencoder=Sequential()
        self.autoencoder.add(firstLayer)
        self.autoencoder.add(AveragePooling1D(pool_length=2, stride=None, border_mode="valid"))
        self.autoencoder.add(Convolution1D(64, 64, activation='relu', border_mode='same',name='conv2'))
        self.autoencoder.add(Convolution1D(64, 64, activation='relu', border_mode='same',name='conv3'))
        self.autoencoder.add(Convolution1D(64,64, activation='relu', border_mode='same',name='conv4'))
        self.autoencoder.add(UpSampling1D(length=2))
        self.autoencoder.add(Convolution1D(64,64, activation='relu', border_mode='same',name='conv5'))
        self.autoencoder.add(Convolution1D(1, 64, activation='relu',border_mode='same'))
        
        self.autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

        if target is None:
            target=self.signals
        if train:
            print 'Training..'
            self.autoencoder.fit(self.signals, target, nb_epoch=15, batch_size=128, shuffle=True, callbacks=[])
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
        return self.autoencoder
