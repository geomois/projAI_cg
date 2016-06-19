from keras.layers import Input, Dense, Convolution1D, UpSampling1D, AveragePooling1D, MaxPooling1D
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
import numpy as np

class kerasModel:
    def __init__(self,waves,rate,annotations):
        self.data = []
        self.signals=waves
        self.rate=rate
        self.annotations=annotations

    def buildAutoEncoder(self,train,target=None):
        #input_au = Input(shape=(1,22050))
        input_au=Input(shape=(22050,1))
        x = Convolution1D(32, 2, activation='relu', border_mode='same')(input_au)#16
        x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
        x = Convolution1D(32, 2, activation='relu', border_mode='same')(x)#16
        encoded = Convolution1D(8, 2, activation='relu', border_mode='same')(x)#8
        encoder = Model(input=input_au, output=encoded)

        x = Convolution1D(8, 2, activation='relu', border_mode='same')(encoded)#8
        x = UpSampling1D(length=2)(x)
        x = Convolution1D(16, 2, activation='relu', border_mode='same')(x)#16
        decoded = Convolution1D(1, 2, activation='relu',border_mode='same')(x)#28
        autoencoder = Model(input_au, decoded)
        autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

        if target is None:
            target=self.signals
        if train:
            print 'Training..'
            autoencoder.fit(self.signals, target, nb_epoch=15, batch_size=128, shuffle=True, callbacks=[])
        autoencoder.save_weights("aE_weights.w", True)
        predictions = autoencoder.predict_on_batch(target)
        error = mean_squared_error(np.resize(self.signals, (len(self.signals), self.rate[0])), np.resize(predictions, (len(predictions), self.rate[0])))
        print 'error ', error
