import keras


class kerasModel():
    def __init__(self,waves,rate):
        self.data = []
#        self.signals=signal.resample(waves,rate)
        self.rate=rate

    def buildModel(self,train):
        model = keras.Sequential()
        model.add(keras.Convolution1D(32, 32, border_mode='same', activation="tanh", input_shape=(self.rate, 1)))
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
