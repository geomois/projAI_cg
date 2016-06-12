import numpy as np
import soundfile as sf
import os
import sys
from model import kerasModel
import scipy.signal as signal

def prepareAudio(directory):
    oggs=[]
    for subdir, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith('.ogg'):
            # if (f.endswith('.ogg') or f.endswith('.mp3')):
            # lib soundfile doesn't recognize .mp3 files
                oggs.append(os.path.join(subdir,f))
    
    oggs=sorted(oggs)
    trainWaves=[]
#    trainRates=[]
    trainRate=0
    stop=1
    if (len(oggs) > 0):
        count = 0
        for path in oggs:
            audioTemp,rateTemp=sf.read(path)
            trainWaves.append(audioTemp)
            trainRate=rateTemp
            count+=1
            if count==stop:
                break
#            trainRates.append(rateTemp)
            # In case we want to save the wave files, uncomment the following line
            # sf.write(os.path.dirname(path)+"/"+os.path.basename(path).split('.')[0]+'.wav',audioTemp,rateTemp)

    return trainWaves,trainRate,oggs

def prepareAnnotation(directory,audioPaths):
    fileNames=[]
    for path in audioPaths:
        fileNames.append(os.path.basename(path).split('.')[0])
        
    timings=[]
    for subdir, dirs, files in os.walk(directory):
        for f in files:
            if f.split('.')[0] in fileNames:
                 timings.append(os.path.join(subdir,f))
    timings=sorted(timings)
    
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
        newRate=(rate*percentage)/100
        resampledSignals.append(np.asarray(signal.resample(waves[i],(len(waves[i])/rate)*newRate)))
        resampledRates.append(newRate)
        
    return resampledSignals,resampledRates

if __name__ == '__main__':
#    waves,rate,paths=prepareAudio(sys.argv[1])
#    annotations=prepareAnnotation(sys.argv[2])
    waves,rate,paths=prepareAudio("/home/george/Desktop/Project AI/projGit/Annotated_music/train/")
    annotations=prepareAnnotation("/home/george/Desktop/Project AI/projGit/Annotated_music/jamendo_lab/",paths)
    signals,downRate=downSample(waves,rate)
#TODO: kalw model
    m=kerasModel(signals,downRate,annotations)
    m.prepareAnnotations()