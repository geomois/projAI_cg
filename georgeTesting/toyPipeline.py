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
#		print "audio duration ", self.audio_duration#, " real duration ", str(984)

		self.chunk_duration=1 # in seconds
#		print "arxika audio shape ", audio.shape
#		print "annot:",annotation[0:10]," ", annotation.shape		

	def split(self):
        	samples_in_chunk=self.audio_sample_freq#WE NOW RECEIVE THE ALREADY DOWNSAMPLED(and since we split in 1 sec it's that rate)
#		print "samples in chunk ", samples_in_chunk
		
        	c=1
		self.chunks=[] #chunks(seconds) in 1st channel
		self.chunks2=[] #chunks in 2nd channel
		self.chunks3=[] #chunks in annotations

        	for start_pos in range(0,self.audio.shape[0],samples_in_chunk):
                	end_pos = start_pos + samples_in_chunk
#               	print "end ",end_pos
			if end_pos >= self.audio.shape[0]:
#				print ">="
                        	end_pos = self.audio.shape[0]- 1
#				print "now ",end_pos 
               		#print "start_pos, end_pos:",start_pos, " ",end_pos
			chunk = self.audio[start_pos : end_pos,0]
			chunk2 = self.audio[start_pos : end_pos,1]
			chunk3 = self.annotation[start_pos : end_pos]
#			print "chunk length ",len(chunk)
                	if len(chunk)<samples_in_chunk:
				chunk=np.append(chunk,np.zeros(samples_in_chunk-len(chunk)))
#				print "now ch",len(chunk)
				chunk2=np.append(chunk2,np.zeros(samples_in_chunk-len(chunk2)))
				chunk3=np.append(chunk3,np.zeros(samples_in_chunk-len(chunk3)))
                        	#print "append some zeros ",len(chunk)
#			print "chunk lengths:",chunk.shape,chunk2.shape,chunk3.shape
                	self.chunks.append(chunk)
			self.chunks2.append(chunk2)
			self.chunks3.append(chunk3)
#		print "chunk lengths:",len(self.chunks),len(self.chunks2),len(self.chunks3)

	def getInputOutputMatrices(self):
		matrix_width=len(self.chunks[0])#afterwards we can resize according for pooling
		matrix_height=len(self.chunks)
		numchannels=2
#		print "matrix_width",matrix_width
#		print "matrix_height",matrix_height
		self.input_matrix=np.zeros(shape=(numchannels,matrix_height,matrix_width))
		self.output_matrix=np.zeros(shape=(matrix_height,matrix_width))
        	for i,chunk in enumerate(self.chunks):
	                self.input_matrix[0,i,:]=self.chunks[i][0:matrix_width]
			self.input_matrix[1,i,:]=self.chunks2[i][0:matrix_width]
       			self.output_matrix[i,:]=self.chunks3[i][0:matrix_width]
#		print"chunks[i][0:matrix_width]:",self.chunks[2][0:matrix_width].shape
#		print"matrix_width",matrix_width
#		print "inp matr shap ",self.input_matrix.shape
#		print "outp matr shap ",self.output_matrix.shape
        	return self.input_matrix,self.output_matrix


#call:
#test=MyAudio(downRate[i],signals[i],1,annotationWave[i])
#test.split()
#inout=test.getInputMatrix()
#test_input_matrix.dstack(inout[0])
#test_output_matrix.dstack(inout[1])
