import numpy as np

class NormAudio():
	
	def __init__(self,signal):
		self.signal=signal
	def normalize(self):
		maxAmp=np.max(np.abs(self.signal))
		newsignal=self.signal/maxAmp
		return newsignal,maxAmp

class RepairAudio():
	def __init__(self,signal,amp):
                self.signal=signal
		self.amp=amp
        def denormalize(self):
                newsignal=self.signal*self.amp
		return newsignal

# USE :
"""
init=np.random.random((1,5,1))
print init
normaudio,amp=NormAudio(init).normalize()
print normaudio,amp
repair=RepairAudio(normaudio,amp).denormalize()
print repair
"""		
