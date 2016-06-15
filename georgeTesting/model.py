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

    def buildAutoEncoder(self,train,):
        #input_au = Input(shape=(1,22050))
        input_au=Input(shape=(22050,1))
        x = Convolution1D(32, 2, activation='relu', border_mode='same')(input_au)#16
        x = AveragePooling1D(pool_length=2, stride=None, border_mode="valid")(x)
        x = Convolution1D(32, 2, activation='relu', border_mode='same')(x)#16
        encoded = Convolution1D(8, 2, activation='relu', border_mode='same')(x)#8
        encoder = Model(input=input_au,output=encoded)
        x = Convolution1D(8, 2, activation='relu', border_mode='same')(encoded)#8
        x=UpSampling1D(length=2)(x)
        x = Convolution1D(16, 2, activation='relu', border_mode='same')(x)#16
        decoded = Convolution1D(1, 2, activation='relu',border_mode='same')(x)#28
        autoencoder = Model(input_au, decoded)
        autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
        
        if train:
            print 'Training..'
            autoencoder.fit(self.signals,self.signals,nb_epoch=15,batch_size=128,shuffle=True,callbacks=[])
        autoencoder.save_weights("aE_weigths.w", True)
        predictions = autoencoder.predict_on_batch(self.signals)
        error = mean_squared_error(np.resize(self.signals, (len(self.signals), self.rate[0])), np.resize(predictions, (len(predictions), self.rate[0])))
        print 'error ', error
#import pickle
##c=pickle.load(open("/home/george/Desktop/Project AI/projGit/georgeTesting/ma.pi",'rb'))
#c=pickle.load(open("/home/gms590/git/projAI_cg/georgeTesting/ma.pi",'rb'))
#bo=c[0]
#b=bo.reshape(len(bo),1,len(bo[0]))
#b=bo
    
#    def buildModel(self,train):
#        model = Sequential()
#        model.add(Convolution1D(32, 32, border_mode='same', activation="tanh", input_shape=(self.rate[0],1)))
#        model.add(AveragePooling1D(pool_length=2, stride=None, border_mode="valid"))
#        model.add(Convolution1D(32, 32, border_mode='same', activation="tanh"))
#        model.add(AveragePooling1D(pool_length=2, stride=None, border_mode="valid"))
#        model.add(Convolution1D(32, 16, border_mode='same', activation="tanh"))
#        model.add(AveragePooling1D(pool_length=2, stride=None, border_mode="valid"))
#        model.add(Convolution1D(1, 8, border_mode='same', activation="tanh"))
#
#        model.add(UpSampling1D(length=2))
#        model.add(Convolution1D(32, 32, border_mode='same', activation="tanh"))
#        model.add(UpSampling1D(length=2))
#        model.add(Convolution1D(32, 32, border_mode='same', activation="tanh"))
#        model.add(UpSampling1D(length=2))
#        model.add(Convolution1D(32, 32, border_mode='same', activation="tanh"))
#        model.add(Convolution1D(1, 32, border_mode='same', activation="tanh"))
#
#        model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
#        if train:
#            print("NOW FITTING")
#            model.fit(self.signals, self.signals, nb_epoch=5000, batch_size=64)
#            model.save_weights("weights_1.dat", True)
#
#        model.load_weights("weights_1.dat")


#        predictions = model.predict_on_batch(x)
#        error = mean_squared_error(np.resize(x, (len(x), sample_rate)), np.resize(predictions, (len(predictions), sample_rate)))
#        print("Train Error: %.4f" % error)
#        for i in range(len(predictions)):
#            prediction = np.resize(predictions[i], (sample_rate,))
#            unnormal = prediction * max
#            unnormal = unnormal.astype(np.int16)
#            wavfile.write("train_predictions/prediction_%d.wav" % (indices[i+100]+1), sample_rate, unnormal)
#
#
#        predictions = model.predict_on_batch(y)
#        error = mean_squared_error(np.resize(y, (len(y), sample_rate)), np.resize(predictions, (len(predictions), sample_rate)))
#        print("Test Error: %.4f" % error)
