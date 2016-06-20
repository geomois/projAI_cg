from keras import backend as K
from keras.layers import Input, Convolution1D, UpSampling1D, AveragePooling1D, MaxPooling1D
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
import numpy as np

placeholder=K.placeholder((1,))
grads = K.gradients(loss, combination_image)
class styleTransfer:
    def __init__(self, trainSignal=None, trainRate=None, trainAnnot=None, validSignal=None, validAnnot=None, validRate=None, contentSignal=None, styleSignal=None):
        self.trainSignal=trainSignal
        self.trainRate=trainRate
        self.trainAnnot=trainAnnot
        self.validSignal=validSignal
        self.validAnnot=validAnnot
        self.validRate=validRate
        self.netModel=None
        
    def buildModel(self):
        input_au=Input(shape=(self.signals.shape[1], 1))
        x = Convolution1D(32, 2, activation='relu', border_mode='same')(input_au)#16
        x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
        x = Convolution1D(32, 2, activation='relu', border_mode='same')(x)#16
        encoded = Convolution1D(8, 2, activation='relu', border_mode='same')(x)#8
        encoder = Model(input=input_au, output=encoded)

        x = Convolution1D(8, 2, activation='relu', border_mode='same')(encoded)#8
        x = UpSampling1D(length=2)(x)
        x = Convolution1D(16, 2, activation='relu', border_mode='same')(x)#16
        decoded = Convolution1D(1, 2, activation='relu',border_mode='same')(x)#28
        self.netModel = Model(input_au, decoded)
        self.netModel.compile(optimizer='adadelta', loss='mean_squared_error')
    def getGram(self):
    def getLoss(self):