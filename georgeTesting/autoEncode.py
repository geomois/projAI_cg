import numpy as np
import soundfile as sf
import os
import sys
#from model import kerasModel
import scipy.signal as signal

def prepareAudio(directory):
    oggs=[]
    for subdir, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith('.ogg'):
            # if (f.endswith('.ogg') or f.endswith('.mp3')):
            # lib soundfile doesn't recognize .mp3 files
                oggs.append(os.path.join(subdir,f))

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

    return trainWaves,trainRate

def downSample(waves,rate):
#   Downsample waveform by percentage of the original signal, fourier transformation
    percentage=50
    newRate=(rate*percentage)/100
    for wave in waves:
        resampledSignals=signal.resample(wave,(len(wave)/rate)*newRate)
        
    return resampledSignals,newRate

if __name__ == '__main__':
#    waves=prepareAudio(sys.argv[1])
    waves,rate=prepareAudio("/home/george/Desktop/Project AI/projAI_cg/Annotated_music/train/")
    signals,downRate=downSample(waves,rate)
#TODO: kalw model