import numpy as np
import soundfile as sf
import time
#utlis
#def waveToFile(waves)

#annot->the outcome of the network annot.shape should be: shape=(#chunks, sample rate)
#rate->the sample rate of the net. (self.rate in model)
#count->the position of the last chuck of each wave
#def annotationsToFile(annot,rate,count):
#    for i in range(len(annot)):
#        with open('annotation'+str(i)+'Out'+time.strftime('%H%M%S').txt,'wb') as f:
#            if i==0: 
#                previous=0
#            else:
#                previous=count(i-1)
#            for k in range(previous,count(i)):
#                temp=annot[previous:k+1,:]
#                temp=temp.reshape(1,(k+1 - previous)*rate[0])
                
                