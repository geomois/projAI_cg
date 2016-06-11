import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import fft, arange
from scipy.signal import spectrogram

"""
#make sine wave
amplitude = 300
time = 5
samplerate = 4000
hz = [400,420,440]
wave = np.zeros(time * samplerate)
for freq in hz:
	wave += np.sin(2 * np.pi * freq * np.linspace(0, time, time  * samplerate))
wave = amplitude * wave / len(hz)
#t=np.linspace(0,time,len(wave))
#plt.plot(t, wave[0:len(wave)])
#plt.show()
"""

def plot_spectrum(filename):
	samplerate,wave=wavfile.read(filename)

	n = len(wave)  # length of the signal
	k = arange(n)
	print n,samplerate
	T = n / samplerate # time step
	frq = k / T  # two sides frequency range
	frq = frq[range(n / 2)]  # one side frequency range
	myfft = fft(wave)
	#print myfft
	normfftwave = fft(wave) / n  # fft computing and normalization
	normfftwave  = normfftwave[range(n / 2)]
	
	plt.figure()
	plt.plot(frq, abs(normfftwave), 'r')  # plotting the spectrum
	plt.xlabel('Freq (Hz)')
	plt.ylabel('Freq (Hz)')
	plt.show()

plot_spectrum('1.wav')

