import numpy as np

def normalize(signal):
	maxAmp=np.max(np.abs(signal))
	newsignal=signal/maxAmp
	return newsignal,maxAmp

def denormalize(signal,amp):
	newsignal=signal*amp
	return newsignal

def chunkIt(signal,rate):
    chunks=[]
    length=len(signal)/rate
    chunks.append(signal[:rate])
    for i in range(1,length):
        chunks.append(signal[i*rate:(i+1)*rate])

    return chunks
