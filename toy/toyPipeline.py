import numpy as np
from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt

class MyAudio:

	def __init__(self,audio_sample_freq,audio,chunk_duration):
		self.audio_sample_freq=audio_sample_freq
		self.audio=audio

	def downsample(self):
        	double_b=440*2
        	#the audio has 441000 samples per sec but we want to downsample
        	#i.e. to have 880 samples per sec so we will decimate every 441000/880

        	step=self.audio_sample_freq/double_b
        	self.downsampled_audio=signal.decimate(self.audio,step)
		self.downsampled_audio=self.downsampled_audio[0:880000]
		self.new_sample_rate=double_b

		maxAmp=np.max(np.abs(self.downsampled_audio))
		self.downsampled_audio=self.downsampled_audio/maxAmp

		print "after normal...min,max=",np.min(self.downsampled_audio),np.max(self.downsampled_audio)
		self.audio=self.downsampled_audio
        	return self.downsampled_audio,self.new_sample_rate,maxAmp

	def split(self):
        	samples_in_chunk=self.new_sample_rate

        	c=1
		self.chunks=[]
		
        	for start_pos in range(0,self.audio.shape[0],samples_in_chunk):
                	end_pos = start_pos + samples_in_chunk
                	if end_pos >= self.audio.shape[0]:
                        	end_pos = self.audio.shape[0] - 1
                	chunk = self.audio[start_pos : end_pos]
                	if chunk.shape[0]<samples_in_chunk:
                        	chunk=np.append(chunk,np.zeros(samples_in_chunk-len(chunk)))#make for channels
                	self.chunks.append(chunk)
        	return self.chunks

	def getInputMatrix(self):
		matrix_width=len(self.chunks[0])#afterwards we can resize according for pooling
		matrix_height=len(self.chunks)
		numchannels=1
		self.input_matrix=np.zeros(shape=(matrix_height,matrix_width))
        	for i,chunk in enumerate(self.chunks):
	                self.input_matrix[i,:]=self.chunks[i][0:matrix_width]
       		print "input matrix shape ",self.input_matrix.shape
        	return self.input_matrix
