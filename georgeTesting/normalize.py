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
    length=signal[0].shape[1]/rate
    temp=signal[0]
    chunks.append(temp[:rate])
    for i in range(1,length):
        t=temp[0][i*rate:(i+1)*rate]
        chunks.append(t.reshape(1,len(t),1))

    return chunks
