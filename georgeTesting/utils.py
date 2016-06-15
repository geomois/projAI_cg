import numpy as np
import soundfile as sf
import time
#utlis
#def waveToFile(waves)

def annotationsToFile(annot,rate):
    for i in range(len(annot)):
        with open('annotation'++time.strftime('%H%M%S').txt,'wb') as f:
            