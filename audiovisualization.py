
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
import numpy as np

data=wavfile.read('1.wav')[1][0:880]
print data.shape

Fs=50 #sampling frequency
T=1/Fs #sampling period
L=880 #length of signal 
t=np.linspace(0,L-1,L)*T #time vector

f=fft(data)
print f.shape
print t.shape


