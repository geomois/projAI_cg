import numpy as np
import soundfile as sf
import os
import sys
from george_model import kerasModel
import scipy.signal as signal
import matplotlib.pylab as plt
import pickle

def prepareAudio(directory,writeFiles):
	oggs=[]
	for subdir, dirs, files in os.walk(directory):
		for f in files:
            		if f.endswith('.ogg'):
				oggs.append(os.path.join(subdir,f))
    	oggs=sorted(oggs)
#	print "oggs ", oggs[0:5]

    	trainWaves=[]
    	trainRates=[]
	trinRate=0
    	stop=5 #how many audio files to preprocess
    	if (len(oggs) > 0):
        	count = 0
        	for path in oggs:
            		audioTemp,rateTemp=sf.read(path)
            		trainWaves.append(audioTemp)
            		trainRate=rateTemp
            		count+=1
            		trainRates.append(rateTemp)
			if writeFiles==True:
            			sf.write(os.path.dirname(path)+"/"+os.path.basename(path).split('.')[0]+'.wav',audioTemp,rateTemp)
			if count==1:
				sf.write("new.wav",audioTemp,rateTemp)
				print "wrote ", path
			if count==stop:
                                break
#	print "trwav: ", trainWaves[0:5]
	print "trRat: ", trainRates[0:5]
    	return trainWaves,trainRate,oggs

def prepareAnnotation(directory,audioPaths):
	fileNames=[]
	for path in audioPaths:
        	fileNames.append(os.path.basename(path).split('.')[0])
#       print "fileNames: ",fileNames[0:5]
    	timings=[]
    	for subdir, dirs, files in os.walk(directory):
        	for f in files:
            		if f.split('.')[0] in fileNames:
                 		timings.append(os.path.join(subdir,f))
    	timings=sorted(timings)
#	print "timings: ",timings[0:5]    

    	annotations=[]
    	for path in timings:
        	with open(path,'r') as aFile:
            		content=aFile.read().splitlines()
            		l=[]            
            		for line in content:
                		temp=line.split(' ')
                		l.append([float(temp[0]), float(temp[1]),(True if temp[2]=='sing' else False)])
            		annotations.append(np.asarray(l))
#    	print "annot: ",annotations[0:5]
    	return annotations

def downSample(waves,rate):
#   	Downsample waveform by percentage of the original signal, fourier transformation
	percentage=50
	resampledSignals=[]
	resampledRates=[]
    	for i in range(0,len(waves)):
		print "downsampling wave ", i
        	newRate=(rate*percentage)/100
        	resampledSignals.append(np.asarray(signal.resample(waves[i],(len(waves[i])/rate)*newRate)))
        	resampledRates.append(newRate)
        	if i==1:
			sf.write("newresampled.wav",waves[i],newRate)
    	return resampledSignals,resampledRates


def visualize(wave,rate):
	if wave.dtype == "float64":
	        w = wave / (2.**63)
        else:
		w = wave / (2**15)
	ws1 =w[:,0]
	ws2 =w[:,1]
        timeArray = plt.arange(0, float(len(ws1)), 1)
        timeArray = timeArray / rate
        timeArray = timeArray * 1000  #scale to milliseconds
        plt.plot(timeArray, ws1, 'r', timeArray, ws2,'b')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (ms)')
        plt.show()


if __name__ == '__main__':
#    waves,rate,paths=prepareAudio("/home/george/Desktop/Project AI/projGit/Annotated_music/train/")
#    annotations=prepareAnnotation("/home/george/Desktop/Project AI/projGit/Annotated_music/jamendo_lab/",paths)

	toload=True
	waves,rate,paths=prepareAudio("train",False)
        annotations=prepareAnnotation("jamendo_lab",paths)
        waves=waves[0:3]#use a sample
	annotations=annotations[0:3]

	if toload==False:
		signals,downRate=downSample(waves,rate)
		pickle.dump((signals,downRate),open( "foursignals.p", "wb" ))
	else:
		signals,downRate = pickle.load(open("foursignals.p","rb"))

	print "sddsdsg:",waves[2].shape," ",rate," ",signals[2].shape," ",downRate
#    	m=kerasModel(signals,downRate,annotations)
#	m.prepareAnnotations()
	
	visualize(waves[2],rate)
        #visualize(signals[2])

