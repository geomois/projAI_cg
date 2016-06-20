from keras import backend as K
from keras.layers import Input, Convolution1D, UpSampling1D, AveragePooling1D, MaxPooling1D
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
import numpy as np
from model import kerasModel
from scipy.optimize import fmin_l_bfgs_b
from scipy.io import wavfile


grads = K.gradients(loss, combination_image)
class styleTransfer:
    def __init__(self, model, sampleRate, contentSignal, styleSignal):
        self.sampleRate=sampleRate 
        self.lValue=None
        self.gradValue=None
        self.content_w=0.025
        self.style_w=1.0        
#        self.trainSignal=trainSignal
#        self.trainRate=trainRate
#        self.trainAnnot=trainAnnot
#        self.validSignal=validSignal
#        self.validAnnot=validAnnot
#        self.validRate=validRate
        self.contentSignal=K.variable(shapeArray(contentSignal))
        self.styleSignal=K.variable(shapeArray(styleSignal))
        self.placeholder=K.placeholder(self.contentSignal.shape)
        self.inputTensor=K.concatenate([self.contentSignal,self.styleSignal,self.placeholder],axis=0)
        self.kModel=model# this is the object, not the model
#        self.output=None
        self.input_au=Input(shape=(3,None,1))#den eimai sigouros
        print '1__'
        self.buildModel()
    
    def shapeArray(ar):
        return ar
        
    def buildModel(self):
        netModel.build(train=False,input_au,None)
        self.netModel=kModel.getModel()
        self.output=dict([(layer.name, layer.output) for layer in netModel.layers])        
        print '2__'        
        loss=K.variable(0.0)
        fMap=outputs_dict['conv1']
        contentFMap=fMap[0,:,:]
        placeholderFMap=fMap[2,:,:]
        loss+=self.content_w*contentLoss(contentFMap,placeholderFMap)
        
        fLayers=['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        for l in fLayers:
            fMap=self.output[l]
            styleFMap=fMap[1,:,:]
            placeholderFMap=fMap[2,:,:]
            styleL=styleLoss(styleFMap,placeholderFMap)
            loss+=(self.style_w/len(fLayers))*styleL
        
        gradients=K.gradients(loss,self.placeholder)
        outGradients=[loss]
#        den 3erw ti paizei edw opote paei comment
#        if type(grads) in {list, tuple}:
#            outputs += grads
#        else:
#            outputs.append(grads)
        
        f_outputs = K.function([self.placeholder], outGradients)
        
    def evaluation(self,x):
        x = x.reshape((1,countSamples,1))
        outs = f_outputs([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values
    
    def getLossValue(self,x):
        self.lValue,self.gradValue=self.evaluation(x)
        return self.lValue
        
    def getGradValue(self,x):
        self.lValue=None
        temp=np.copy(self.gradValue)
        self.gradValue=None
        return temp        
        
    def getGram(self,matrix):
        assert K.ndim(matrix) == 2 , "gram ndim not 2"
        gram=K.dot(matrix.T,matrix)
        return gram
#        features = K.batch_flatten(x)
#        gram = K.dot(features, K.transpose(features))
        
    def styleLoss(self,style,placehold):
        assert K.ndim(style)==2 ,"style ndim not 2"
        assert K.ndim(placehold==2), "placeholder ndim not 2"
#        predStyle=self.netModel.predict(self.styleSignal) #feature maps
        Sg=self.getGram(style)
        Pg=self.getGram(placehold)
        return K.sum(K.square(Sg-Pg))/ K.sum(K.square(predStyle)) #may be Sg-Pg       
        
    def contentLoss(self,content,placehold):
        Pg=self.getGram(self.placeholder)
        Cg=self.getGram(self.netModel.predict(self.contentSignal))
        loss=K.sum(K.square(Pg-Cg))
        return loss            
        
    def run(self,variousFlags=None):
        x = self.shapeArray(np.random.random((1,len(self.contentSignal),1)))
        for i in range(10):
            print('Start of iteration', i)
            start_time = time.time()
            x, min_val, info = fmin_l_bfgs_b(self.getLossValue, x.flatten(),
                                         fprime=self.getGradValue, maxfun=20)
            print('Current loss value:', min_val)
            # save current generated image
            img = deprocess_image(x.reshape((3, img_width, img_height)))
            wavfile.write('outFiles/output.wav', self.sampleRate, y.astype(np.int16))
            end_time = time.time()
            print('Image saved as', fname)
            print('Iteration %d completed in %ds' % (i, end_time - start_time))
                