import keras
import numpy as np

class kerasModel:
    def __init__(self,waves,rate,annotations):
        self.data = []
        self.signals=waves
        self.rate=rate
        self.annotations=annotations
        
#    def prepareTrainData(aWaves,rate):
        
    def prepareAnnotations(self):
        annotations=self.annotations
        aWaves=[]
        for wave in self.signals:
            aWaveTemp=np.zeros((1,len(wave)))
            for i in range(0,len(self.rate)):
                for j in range(0,len(annotations[i])):
                    if(annotations[i][j][2]): # True -> sing -> 1
                        start=np.ceil(self.rate[i]*annotations[i][j][0])
                        end=np.floor(self.rate[i]*annotations[i][j][1])
                        aWaveTemp[0][start:end]=[1 for k in range(int(start),int(end))]
            aWaves.append(aWaveTemp[0])
#        print aWaves[0].shape
        
    def buildModel(self,train):
        model = keras.Sequential()
        model.add(keras.Convolution1D(32, 32, border_mode='same', activation="tanh", input_shape=(9, 1)))
        model.add(keras.AveragePooling1D(pool_length=2, stride=None, border_mode="valid"))
        model.add(keras.Convolution1D(32, 32, border_mode='same', activation="tanh"))
        model.add(keras.AveragePooling1D(pool_length=2, stride=None, border_mode="valid"))
        model.add(keras.Convolution1D(32, 16, border_mode='same', activation="tanh"))
        model.add(keras.AveragePooling1D(pool_length=2, stride=None, border_mode="valid"))
        model.add(keras.Convolution1D(1, 8, border_mode='same', activation="tanh"))

        model.add(keras.UpSampling1D(length=2))
        model.add(keras.Convolution1D(32, 32, border_mode='same', activation="tanh"))
        model.add(keras.UpSampling1D(length=2))
        model.add(keras.Convolution1D(32, 32, border_mode='same', activation="tanh"))
        model.add(keras.UpSampling1D(length=2))
        model.add(keras.Convolution1D(32, 32, border_mode='same', activation="tanh"))
        model.add(keras.Convolution1D(1, 32, border_mode='same', activation="tanh"))

        model.compile(loss='mean_squared_error', optimizer=keras.SGD(lr=0.01, momentum=0.9, nesterov=True))
#        if train:
#            print("NOW FITTING")
#            model.fit(x, x, nb_epoch=5000, batch_size=64)
#            model.save_weights("weights_1.dat", True)

        model.load_weights("weights_1.dat")


        predictions = model.predict_on_batch(x)
#        error = keras.mean_squared_error(np.resize(x, (len(x), sample_rate)), np.resize(predictions, (len(predictions), sample_rate)))
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
