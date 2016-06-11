import numpy as np
from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt

class MyAudio:

	def __init__(self,filename,maxFreq,chunk_duration):
		self.audio_sample_freq, self.audio = wavfile.read('toy_data_sines_44_1khz.wav')
		print "audio sample freq ", self.audio_sample_freq

		self.audio_duration=self.audio.shape[0]/self.audio_sample_freq
		print "audio duration ", self.audio_duration, " real duration ", str(984)

		self.maxFreq=440
		self.chunk_duration=1 # in seconds

	def downsample(self):
        	double_b=self.maxFreq*2
        	#the audio has 441000 samples per sec but we want to downsample
        	#i.e. to have 880 samples per sec so we will decimate every 441000/880

        	#test how many channels
        	print "audio shape ", self.audio.shape
        	step=self.audio_sample_freq/double_b
        	print "step ", step
        	self.downsampled_audio=signal.decimate(self.audio,step)
        	return self.downsampled_audio,step

	def split(self,write_files):
        	samples_in_chunk=self.downsampled_audio.shape[0]/self.audio_duration
        	print "samples in chunk ", samples_in_chunk

        	c=1
		self.chunks=[]

        	for start_pos in range(0,self.downsampled_audio.shape[0],samples_in_chunk):
                	end_pos = start_pos + samples_in_chunk
                	if end_pos >= len(self.downsampled_audio):
                        	end_pos = len(self.downsampled_audio) - 1
               		#print "start_pos, end_pos:",start_pos, " ",end_pos
                	chunk = self.downsampled_audio[start_pos : end_pos]
                	if chunk.shape[0]<samples_in_chunk:
                        	chunk=np.append(chunk,np.zeros(samples_in_chunk-len(chunk)))#make for channels
                        	#print "append some zeros ",len(chunk)
                	if write_files==True:
                        	wavfile.write("created/"+str(c)+".wav",samples_in_chunk/chunk_duration, chunk)
                	c = c + 1
                	#print "wrote file: ", str(c)+".wav"
                	self.chunks.append(chunk)
        	return self.chunks

	def getInputMatrix(self):
	        matrix_width=len(self.chunks[0]) #this is 882 but we ll use 880
		matrix_width=880
		matrix_height=len(self.chunks)

		self.input_matrix=np.zeros(shape=(matrix_height,matrix_width))
        	for i,chunk in enumerate(self.chunks):
	                self.input_matrix[i,:]=chunk[0:matrix_width] # fix for channels
       		print self.input_matrix.shape
        	return self.input_matrix

#pipeline:

test=MyAudio('toy_data_sines_44_1khz.wav',440,1)
downsampled_test,sample_freq=test.downsample()
print "downsampled_audio shape,freq ",downsampled_test.shape, " ",sample_freq
#print "sampled audio duration ", downsampled_audio.shape[0]/sample_freq, " BAD"
audio_chunks_of1sec = test.split(False)
input_matrix=test.getInputMatrix()
