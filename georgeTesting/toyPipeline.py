import numpy as np
from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt

class MyAudio:

	def __init__(self,audio_sample_freq,audio,chunk_duration,annotation):
		self.audio_sample_freq=audio_sample_freq
		self.audio=audio
		self.annotation=annotation
		#print "audio sample freq ", self.audio_sample_freq

		self.audio_duration=self.audio.shape[0]/self.audio_sample_freq
		print "audio duration ", self.audio_duration#, " real duration ", str(984)

		self.chunk_duration=1 # in seconds
		print "arxika audio shape ", audio.shape

	def split(self):
        	samples_in_chunk=self.audio_sample_freq#WE NOW RECEIVE THE ALREADY DOWNSAMPLED(and since we split in 1 sec it's that rate)
		print "samples in chunk ", samples_in_chunk
		
        	c=1
		self.chunks=[] #chunks(seconds) in 1st channel
		self.chunks2=[] #chunks in 2nd channel
		self.chunks3=[] #chunks in annotations

        	for start_pos in range(0,self.audio.shape[0],samples_in_chunk):
                	end_pos = start_pos + samples_in_chunk
                	if end_pos >= self.audio.shape[0]:
                        	end_pos = self.audio.shape[0]- 1
               		#print "start_pos, end_pos:",start_pos, " ",end_pos
                	chunk = self.audio[start_pos : end_pos,0]
			chunk2 = self.audio[start_pos : end_pos,1]
			chunk3 = self.annotation[start_pos : end_pos]
                	if chunk.shape[0]<samples_in_chunk:
                        	chunk=np.append(chunk,np.zeros(samples_in_chunk-len(chunk)))#make for channels
				chunk2=np.append(chunk2,np.zeros(samples_in_chunk-len(chunk)))
				chunk3=np.append(chunk3,np.zeros(samples_in_chunk-len(chunk)))
                        	#print "append some zeros ",len(chunk)
                	self.chunks.append(chunk)
			self.chunks2.append(chunk2)
			self.chunks3.append(chunk3)
			
		#toreturn=np.vstack((np.array(self.chunks),np.array(self.chunks2)))
		#print "to return chunks shape ", toreturn.shape

	def getInputOutputMatrices(self):
		matrix_width=len(self.chunks[0])#afterwards we can resize according for pooling
		matrix_height=len(self.chunks)
		numchannels=2
		self.input_matrix=np.zeros(shape=(matrix_height,matrix_width,numchannels))
		self.output_matrix=np.zeros(shape=(matrix_height,matrix_width))
        	for i,chunk in enumerate(self.chunks):
	                self.input_matrix[i,:,0]=self.chunks[i][0:matrix_width]
			self.input_matrix[i,:,1]=self.chunks2[i][0:matrix_width]
       			self.output_matrix[i,:]=self.chunks3[i][0:matrix_width]
		print"chunks[i][0:matrix_width]:",self.chunks[2][0:matrix_width].shape
		print"matrix_width",matrix_width
		print "inp matr shap ",self.input_matrix.shape
		print "outp matr shap ",self.output_matrix.shape
        	return self.input_matrix,self.output_matrix

