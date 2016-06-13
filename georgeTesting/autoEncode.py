import numpy as np
import soundfile as sf
import os
import sys
#from model import kerasModel
import scipy.signal as signal
import pickle

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

def readAnnotations(directory,audioPaths):
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

    
def prepareAnnotations(signals,rate,annotations):
    aWaves=[]
    for wave in signals:
        aWaveTemp=np.zeros((1,len(wave)))
        for i in range(0,len(rate)):
            for j in range(0,len(annotations[i])):
                if(annotations[i][j][2]): # True -> sing -> 1
                    start=np.ceil(rate[i]*annotations[i][j][0])
                    end=np.floor(rate[i]*annotations[i][j][1])
                    aWaveTemp[0][start:end]=[1 for k in range(int(start),int(end))]
        aWaves.append(aWaveTemp[0])
    
    return aWaves

def toPickle(writeFlag,waves=None,rates=None,annotation=None):
    if writeFlag:
        pickle.dump(annotation,open('pickled/annot.pi','wb'))
        pickle.dump(waves,open('pickled/wav.pi','wb'))
        pickle.dump(rates,open('pickled/rates.pi','wb'))
    else:
        waves= pickle.load(open("pickled/wav.pi","rb"))
        annotation=pickle.load(open('pickled/annot.pi','rb'))
        rates=pickle.load(open('pickled/rates.pi','rb'))
        return waves,annotation,rates        
        
if __name__ == '__main__':
    simpleRun=False
    if sys.argv[3] =='read':
        waves,annotation,rates = toPickle(False)
    elif sys.argv[3] =='write':
        waves,rate,paths=prepareAudio(sys.argv[1])
        annotations=readAnnotations(sys.argv[2],paths)
        signals,downRate=downSample(waves,rate)
        annotationWave=prepareAnnotations(signals,downRate,annotations) 
        toPickle(True,signals,downRate,annotationWave)
     else:
         simpleRun=True
         
     if simpleRun:
        waves,rate,paths=prepareAudio(sys.argv[1])
        annotations=readAnnotations(sys.argv[2],paths)
        signals,downRate=downSample(waves,rate)
        annotationWave=prepareAnnotations(signals,downRate,annotations)
#    waves,rate,paths=prepareAudio("/home/george/Desktop/Project AI/projGit/Annotated_music/train/")
#    annotations=prepareAnnotations("/home/george/Desktop/Project AI/projGit/Annotated_music/jamendo_lab/",paths)
#    signals,downRate=downSample(waves,rate)
#    annotationWave=prepareAnnotations(signals,downRate,annotations)
     
##TODO: kalw model
#    m=kerasModel(signals,downRate,annotations)