import numpy as np
from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt

class MyAudio:

	def __init__(self,audio_sample_freq,audio,chunk_duration):
		#self.audio_sample_freq, self.audio = wavfile.read('toy_data_sines_44_1khz.wav')
		self.audio_sample_freq=audio_sample_freq
		self.audio=audio
		#print "audio sample freq ", self.audio_sample_freq

		#self.audio_duration=self.audio.shape[0]/self.audio_sample_freq
		self.audio_duration=self.audio.shape[0]/self.audio_sample_freq
		print "audio duration ", self.audio_duration#, " real duration ", str(984)

		#self.maxFreq=440
		self.chunk_duration=1 # in seconds
		print "arxika audio shape ", audio.shape

	def split(self):
        	#samples_in_chunk=self.downsampled_audio.shape[0]/self.audio_duration
        	samples_in_chunk=self.audio_sample_freq#WE NOW RECEIVE THE ALREADY DOWNSAMPLED(and since we split in 1 sec it's that rate)
		print "samples in chunk ", samples_in_chunk

        	c=1
		self.chunks=[]
		self.chunks2=[]
		
        	for start_pos in range(0,self.audio.shape[0],samples_in_chunk):
                	end_pos = start_pos + samples_in_chunk
                	if end_pos >= self.audio.shape[0]:
                        	end_pos = self.audio.shape[0]- 1
               		#print "start_pos, end_pos:",start_pos, " ",end_pos
                	chunk = self.audio[start_pos : end_pos,0]
			chunk2 = self.audio[start_pos : end_pos,1]
                	if chunk.shape[0]<samples_in_chunk:
                        	chunk=np.append(chunk,np.zeros(samples_in_chunk-len(chunk)))#make for channels
				chunk2=np.append(chunk2,np.zeros(samples_in_chunk-len(chunk2)))
                        	#print "append some zeros ",len(chunk)
                	"""
			if write_files==True:
                        	wavfile.write("created/"+str(c)+".wav",samples_in_chunk/chunk_duration, chunk)
			c = c + 1
                	#print "wrote file: ", str(c)+".wav"
			"""
                	self.chunks.append(chunk)
			self.chunks2.append(chunk2)
			
		toreturn=np.vstack((np.array(self.chunks),np.array(self.chunks2)))
		print "to return chunks shape ", toreturn.shape
        	#return self.chunks

	def getInputMatrix(self):
	        #matrix_width=len(self.chunks[0]) #this is 882 but we ll use 880
		#matrix_width=880

		matrix_width=len(self.chunks[0])#afterwards we can resize according for pooling
		matrix_height=len(self.chunks)
		numchannels=2
		self.input_matrix=np.zeros(shape=(matrix_height,matrix_width,numchannels))
        	for i,chunk in enumerate(self.chunks):
	                self.input_matrix[i,:,0]=self.chunks[i][0:matrix_width]
			self.input_matrix[i,:,1]=self.chunks2[i][0:matrix_width]
       		print "inp matr shap ",self.input_matrix.shape
        	return self.input_matrix

#pipeline:
"""
test=MyAudio('toy_data_sines_44_1khz.wav',440,1)
downsampled_test,sample_freq=test.downsample()
print "downsampled_audio shape,freq ",downsampled_test.shape, " ",sample_freq
#print "sampled audio duration ", downsampled_audio.shape[0]/sample_freq, " BAD"
audio_chunks_of1sec = test.split(False)
input_matrix=test.getInputMatrix()
"""
