#call: python <scriptName> <trainFiles directory> <annotations directory> <read> or <write> or <ogg> or <simple> <number of files to read>

import numpy as np
import soundfile as sf
import os
import sys
from model import kerasModel
import scipy.signal as signal
import pickle
from toyPipeline import MyAudio
import gzip

def prepareAudio(directory,size=1):
    oggs=[]
    for subdir, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith('.ogg'):
            # if (f.endswith('.ogg') or f.endswith('.mp3')):
            # lib soundfile doesn't recognize .mp3 files
                oggs.append(os.path.join(subdir,f))
    
    oggs=sorted(oggs)
    trainWaves=[]
    trainRates=[]
    if size>len(oggs):
        stop=len(oggs)
    else:
        stop=size
        
    if (len(oggs) > 0):
        count = 0
        for path in oggs:
            audioTemp,rateTemp=sf.read(path)
            trainWaves.append(audioTemp)
            count+=1
	    trainRates.append(rateTemp)
            if count==stop:
                break
            # In case we want to save the wave files, uncomment the following line
            # sf.write(os.path.dirname(path)+"/"+os.path.basename(path).split('.')[0]+'.wav',audioTemp,rateTemp)
    return trainWaves,trainRates,oggs

def readAnnotations(directory,audioPaths):
#    fileNames=[]
    timings=[]
    for path in audioPaths:
        timings.append(os.path.dirname(directory)+'/'+os.path.basename(path).split('.')[0]+'.lab')
#	print os.path.basename(os.path.dirname(directory)+'/'+os.path.basename(path).split('.')[0]+'.lab')
    annotations=[]
    for path in timings:
        with open(path,'r') as aFile:
            content=aFile.read().splitlines()
            l=[]            
            for line in content:
                temp=line.split(' ')
                l.append([float(temp[0]), float(temp[1]),(True if temp[2]=='sing' else False)])
            annotations.append(np.asarray(l))
    
    return annotations

def downSample(waves,rate):
#   Downsample waveform by percentage of the original signal, fourier transformation
    percentage=50
    resampledSignals=[]
    resampledRates=[]
    for i in range(0,len(waves)):
        newRate=(rate[i]*percentage)/100
        temp=np.asarray(signal.resample(waves[i],(len(waves[i])/rate[i])*newRate))
        resampledSignals.append(temp)
        print 'reasampled: ',i
        resampledRates.append(newRate)
 
    return resampledSignals,resampledRates

    
def prepareAnnotations(signals,rate,annotations):
    aWaves=[]
    for k in range(0,len(signals)):
        aWaveTemp=np.zeros((1,len(signals[k])))
        for j in range(0,len(annotations[k])):
		 if(annotations[k][j][2]):
			start=np.ceil(rate[k]*annotations[k][j][0])
                        end=np.floor(rate[k]*annotations[k][j][1])
                        aWaveTemp[0][start:end]=[1 for u in range(int(start),int(end))]
        aWaves.append(aWaveTemp[0])
    
    return aWaves

def toPickle(writeFlag,waves=None,rates=None,annotation=None):
    if writeFlag:
        if annotation != None:
            f = gzip.open('pickled/annot.pi.pklz','wb')
            pickle.dump(annotation,f)
            f.close()
        if waves != None:
            f = gzip.open('pickled/wav.pi.pklz','wb')
            pickle.dump(waves,f)
            f.close()
        if rates != None:
            f=gzip.open('pickled/rates.pi.pklz','wb')
            pickle.dump(rates,open('pickled/rates.pi','wb'))
            f.close()
    else:
        if waves != None:
            f=gzip.open('pickled/wav.pi.pklz','rb')
            waves=pickle.load(f)
            f.close()
            waves= pickle.load(open("pickled/wav.pi","rb"))
        if annotation != None:
            f=gzip.open('pickled/annot.pi.pklz','rb')
            annotation=pickle.load(f)
            f.close()
            annotation=pickle.load(open('pickled/annot.pi','rb'))
        if rates != None:
            f=gzip.open('pickled/rates.pi.pklz','rb')
            rates=pickle.load(f)
            f.close()
            
        return waves,annotation,rates        


def toOgg(waves,rates,paths):
    print 'toOgg'
    for i in range(0,len(waves)):
        sf.write('resampled/'+ os.path.basename(paths[i]),waves[i][:,:],rates[i])

if __name__ == '__main__':
#    simpleRun=True

    simpleRun=False
    if sys.argv[3] =='read':
        waves,annotation,rates = toPickle(False,1,1,1)
    elif sys.argv[3] =='write':
        waves,rate,paths=prepareAudio(sys.argv[1])
        annotations=readAnnotations(sys.argv[2],paths)
        signals,downRate=downSample(waves,rate)
        annotationWave=prepareAnnotations(signals,downRate,annotations) 
        toPickle(True,signals,downRate,annotationWave)
    elif sys.argv[3] == 'ogg':
        waves,rate,paths=prepareAudio(sys.argv[1])
        annotations=readAnnotations(sys.argv[2],paths)
        signals,downRate=downSample(waves,rate)
        annotationWave=prepareAnnotations(signals,downRate,annotations)
        toOgg(signals,downRate,paths)
    else:
        simpleRun=True
         
    if simpleRun:
        print 'simpleRun'
        waves,rate,paths=prepareAudio(sys.argv[1],int(sys.argv[4]))
        annotations=readAnnotations(sys.argv[2],paths)
        signals,downRate=downSample(waves,rate)
        annotationWave=prepareAnnotations(signals,downRate,annotations)

#    waves,rate,paths=prepareAudio("/home/george/Desktop/Project AI/projGit/Annotated_music/train/")
#    annotations=readAnnotations("/home/george/Desktop/Project AI/projGit/Annotated_music/jamendo_lab/",paths)
#    signals,downRate=downSample(waves,rate)
#    annotationWave=prepareAnnotations(signals,downRate,annotations)
#     
    #for one audio (i.e. signal in class autoEncode.py) call:

    test=MyAudio(downRate[0],signals[0],1,annotationWave[0])
    test.split()
    inp,out=test.getInputOutputMatrices()
    test_input_matrix=inp
    test_output_matrix=out
    for i in range(1,len(signals)):   
        test=MyAudio(downRate[i],signals[i],1,annotationWave[i])
        test.split()
        inp,out=test.getInputOutputMatrices()
        test_input_matrix=np.hstack((test_input_matrix,inp))
        test_output_matrix=np.vstack((test_output_matrix,out))
        
    
    outArray=test_output_matrix.reshape((test_output_matrix.shape[0],test_output_matrix.shape[1],1))
    inArray=test_input_matrix[0].reshape((test_input_matrix[0].shape[0],test_input_matrix[0].shape[1],1))

    print "outArray ", outArray.shape
    print "inArray ", inArray.shape

    m=kerasModel(inArray,downRate,outArray)
    m.buildAutoEncoder(True)

