from keras.layers import Convolution1D, UpSampling1D, AveragePooling1D, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import SGD
from scipy.io import wavfile
import numpy as np
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt

from toyPipeline import MyAudio

data=MyAudio('toy_data_sines_44_1khz.wav',440,1)
data.downsample()
data.split(False)

data_input=data.getInputMatrix()

print "NOW:",data_input.shape


#model=Sequential()
#model.add(Convolution1D)

#audio visualization:

