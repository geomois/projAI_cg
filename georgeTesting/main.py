# call: python <scriptName> <trainFiles directory> <annotations directory> <read> or <write> or <wav> or <simple> <number of files to read>
import numpy as np
import soundfile as sf
import os
import sys
from kModel import kModel
import scipy.signal as signal
import pickle
from monoPipeline import MyAudio
import gzip
from styleModel import *

def prepareAudio(directory, size=1):
    oggs = []
    for subdir, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith('.ogg') or f.endswith('.wav'):
                # if (f.endswith('.ogg') or f.endswith('.mp3')):
                # lib soundfile doesn't recognize .mp3 files
                oggs.append(os.path.join(subdir, f))

    oggs = sorted(oggs)
    trainWaves = []
    trainRates = []
    if size > len(oggs):
        stop = len(oggs)
    else:
        stop = size

    if (len(oggs) > 0):
        count = 0
        for path in oggs:
            audioTemp, rateTemp = sf.read(path)
            trainWaves.append(audioTemp)
            count += 1
            trainRates.append(rateTemp)
            # In case we want to save the wave files, uncomment the following line
            # sf.write(os.path.dirname(path) + "/" + os.path.basename(path).split('.')[0] + '.wav', audioTemp, rateTemp)
            if count == stop:
                break
    return trainWaves, trainRates, oggs


def readAnnotations(directory, audioPaths):
    #    fileNames=[]
    timings = []
    for path in audioPaths:
        timings.append(os.path.dirname(directory) + '/' + os.path.basename(path).split('.')[0] + '.lab')
    #	print os.path.basename(os.path.dirname(directory)+'/'+os.path.basename(path).split('.')[0]+'.lab')
    annotations = []
    for path in timings:
        with open(path, 'r') as aFile:
            content = aFile.read().splitlines()
            l = []
            for line in content:
                temp = line.split(' ')
                l.append([float(temp[0]), float(temp[1]), (True if temp[2] == 'sing' else False)])
            annotations.append(np.asarray(l))

    return annotations

def toMono(waves):
    mono=[]
    for i in range(len(waves)):
        mono.append(np.sum(waves[i], 1)/2.0)
    print 'mono ', mono[0].shape
    return mono


def downSample(waves, rate,paths):
    #   Downsample waveform by percentage of the original signal, fourier transformation
    percentage = 30
    resampledSignals = []
    resampledRates = []
    for i in range(0, len(waves)):
        newRate = (rate[i] * percentage) / 100
        temp = np.asarray(signal.resample(waves[i], (len(waves[i]) / rate[i]) * newRate))
        resampledSignals.append(temp)
        print 'resampled: ', i
        resampledRates.append(newRate)
        if paths is not None:
            toWav([temp],[newRate],[paths[i]])
    return resampledSignals, resampledRates


def prepareAnnotations(signals, rate, annotations):
    aWaves = []
    for k in range(0, len(signals)):
        aWaveTemp = np.zeros((1, len(signals[k])))
        for j in range(0, len(annotations[k])):
            if (annotations[k][j][2]):
                start = np.ceil(rate[k] * annotations[k][j][0])
                end = np.floor(rate[k] * annotations[k][j][1])
                aWaveTemp[0][start:end] = [1 for u in range(int(start), int(end))]
        aWaves.append(aWaveTemp[0])

    return aWaves

def toPickle(writeFlag, waves=None, rates=None , annotation=None):
    if writeFlag:
        if annotation != None:
            f = gzip.open('/local/gms590/pickled/annot.pi.pklz', 'wb')
            pickle.dump(annotation, f)
            f.close()
        if waves != None:
            f = gzip.open('/local/gms590/pickled/wav.pi.pklz', 'wb')
            pickle.dump(waves, f)
            f.close()
        if rates != None:
            f = gzip.open('/local/gms590/pickled/rates.pi.pklz', 'wb')
            pickle.dump(rates, open('/local/gms590/pickled/rates.pi', 'wb'))
            f.close()
    else:
        if waves is not None:
            f = gzip.open('/local/gms590/pickled/wav.pi.pklz', 'rb')
            waves = pickle.load(f)
            f.close()
            waves = pickle.load(open("/local/gms590/pickled/wav.pi", "rb"))
        if annotation != None:
            f = gzip.open('/local/gms590/pickled/annot.pi.pklz', 'rb')
            annotation = pickle.load(f)
            f.close()
            annotation = pickle.load(open('/local/gms590/pickled/annot.pi', 'rb'))
        if rates != None:
            f = gzip.open('/local/gms590/pickled/rates.pi.pklz', 'rb')
            rates = pickle.load(f)
            f.close()

        return waves, annotation, rates

def toWav(waves, rates, paths):
    print 'toWav'
    for i in range(0, len(waves)):
        sf.write('../resampledValid/'+os.path.basename(paths[i]).split('.')[0]+'.wav', waves[i], rates[i])

def prepareForKeras(downRate,signals,annotationWave):
    # for one audio (i.e. signal in class autoEncode.py) call:

    endFile=[]
    test = MyAudio(downRate[0], signals[0], 1, annotationWave[0])
    test.split()
    inp, out = test.getInputOutputMatrices()
    test_input_matrix = inp
    test_output_matrix = out
    endFile.append(len(test_output_matrix))
    for i in range(1, len(signals)):
        test = MyAudio(downRate[i], signals[i], 1, annotationWave[i])
        test.split()
        inp, out = test.getInputOutputMatrices()
        test_input_matrix = np.hstack((test_input_matrix, inp))
        test_output_matrix = np.vstack((test_output_matrix, out))
        endFile.append(len(test_output_matrix))

    outArray = test_output_matrix.reshape((test_output_matrix.shape[0], test_output_matrix.shape[1], 1))
    inArray = test_input_matrix[0].reshape((test_input_matrix[0].shape[0], test_input_matrix[0].shape[1], 1))

    return outArray,inArray


if __name__ == '__main__':
    simpleRun = False
    if sys.argv[3] == 'read' or sys.argv[3] == 'readValid':
        signals, downRate, paths=prepareAudio('../resampled/', 100)
        annotations = readAnnotations(sys.argv[2], paths)
        annotationWave=prepareAnnotations(signals, downRate, annotations)
        if sys.argv[3] == 'readValid':
            validSignals, validRate, validPaths = prepareAudio('../resampledValid/', 100)
            annotations = readAnnotations(sys.argv[2], validPaths)
            validAnnotWave = prepareAnnotations(validSignals, validRate, annotations)

    elif sys.argv[3] == 'write':
        waves, rate, paths = prepareAudio(sys.argv[1], int(sys.argv[4]))
        annotations = readAnnotations(sys.argv[2], paths)
        signals, downRate = downSample(waves, rate,None)
        annotationWave = prepareAnnotations(signals, downRate, annotations)
        toPickle(True, signals, downRate, annotationWave)
    elif sys.argv[3] == 'wav':
        waves, rate, paths = prepareAudio(sys.argv[1], int(sys.argv[4]))
        annotations = readAnnotations(sys.argv[2], paths)
        waves = toMono(waves)
        signals, downRate = downSample(waves, rate, paths)
        # length=len(waves)
        # signals=[]
        # for g in range(0,length,2):
        #     tempWaves=waves[0:2]
        #     tempSignals, downRate = downSample(tempWaves, rate)
        #     signals.extend(tempSignals)
        #     del waves[0:2]

        annotationWave = prepareAnnotations(signals, downRate, annotations)
        toWav(signals, downRate, paths)
        print 'done saving'
    else:
        simpleRun = True

    if simpleRun:
        print 'simpleRun'
        waves, rate, paths = prepareAudio(sys.argv[1], int(sys.argv[4]))
        annotations = readAnnotations(sys.argv[2], paths)
        signals, downRate = downSample(waves, rate, None)
        annotationWave = prepareAnnotations(signals, downRate, annotations)

    annotArray,sigArray=prepareForKeras(downRate,signals,annotationWave)
    annotValid,sigValid=prepareForKeras(validRate,validSignals,validAnnotWave)
    
    print "annotTrain ", annotArray.shape
    print "sigTrain ", sigArray.shape
    print 'annotValid ', annotValid.shape
    print 'sigValid ', sigValid.shape
    batch=100
    m = kModel(sigArray, downRate,annotArray,sigValid[:batch],annotValid[:batch],validRate)
    # sT=styleTransfer(m,downRate[0],sigArray[:326],sigArray[326:652])
    print 'initiated'
    start(m,downRate[0],sigArray[:326],sigArray[326:652])
    # sT.run()
#    m.buildAutoEncoder(True, annotArray)
#    m.predict()
