import numpy as np
from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt

audio_sample_freq, audio = wavfile.read('toy_data_sines_44_1khz.wav')
print "audio sample freq ", audio_sample_freq

audio_duration=audio.shape[0]/audio_sample_freq
print "audio duration ", audio_duration, " real	duration ", str(984)

maxFreq=440

def downsample(audio,audio_sample_freq,maxFreq):
	double_b=maxFreq*2
	#the audio has 441000 samples per sec but we want to downsample
	#i.e. to have 880 samples per sec so we will decimate every 441000/880

	#test how many channels
	print "audio shape ", audio.shape
	step=audio_sample_freq/double_b
	print "step ", step
	downsampled_audio=signal.decimate(audio,step)
	return downsampled_audio,step

downsampled_audio,sample_freq=downsample(audio,audio_sample_freq,maxFreq)
print "downsampled_audio shape ",downsampled_audio.shape
print "downsampled_audio frequency ",sample_freq

print "sampled audio duration ", downsampled_audio.shape[0]/sample_freq, " BAD"
 
chunk_duration=1 # in seconds

def split(audio,chunk_duration,sample_freq,audio_duration,write_files):
	samples_in_chunk=chunk_duration*sample_freq
	print "samples in chunk ", samples_in_chunk
	samples_in_chunk=audio.shape[0]/audio_duration
	print "samples in chunk ", samples_in_chunk

	c=1
	chunks=[]

	for start_pos in range(0,audio.shape[0],samples_in_chunk):
        	end_pos = start_pos + samples_in_chunk 
		if end_pos >= len(audio):
			end_pos = len(audio) - 1
		#print "start_pos, end_pos:",start_pos, " ",end_pos 
        	chunk = audio[start_pos : end_pos]
		if chunk.shape[0]<samples_in_chunk:
			chunk=np.append(chunk,np.zeros(samples_in_chunk-len(chunk)))#should make it work for many channels
			#print "append some zeros ",len(chunk)
		if write_files==True:
			wavfile.write("created/"+str(c)+".wav",samples_in_chunk/chunk_duration,chunk)
		c = c + 1
		#print "wrote file: ", str(c)+".wav"
		chunks.append(chunk)
	return chunks

audio_chunks_of1sec = split(downsampled_audio,chunk_duration,sample_freq,audio_duration,False)

matrix_width=len(audio_chunks_of1sec[0]) #this is 882...but we will use 880 which can be divided by 2 many times (for CNN pooling)
matrix_width=880
matrix_height=len(audio_chunks_of1sec)

def getInputMatrix(matrix_width,matrix_height,audio_chunks_of_1sec):
	input_matrix=np.zeros(shape=(matrix_height,matrix_width))
	for i,chunk in enumerate(audio_chunks_of1sec):
		input_matrix[i,:]=chunk[0:matrix_width] # should make it work for multiple channels

	print input_matrix.shape
	return input_matrix

input_matrix=getInputMatrix(matrix_width,matrix_height,audio_chunks_of1sec)

