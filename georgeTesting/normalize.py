import numpy as np

def normalize(signal):
	maxAmp=np.max(np.abs(signal))
	newsignal=signal/maxAmp
	return newsignal,maxAmp

def denormalize(signal,amp):
	newsignal=signal*amp
	return newsignal

# USE :
"""
init=np.random.random((1,5,1))
print init
normaudio,amp=NormAudio(init).normalize()
print normaudio,amp
repair=RepairAudio(normaudio,amp).denormalize()
print repair
"""		
