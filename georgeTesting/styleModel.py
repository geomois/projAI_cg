from keras import backend as K
import numpy as np
from keras.layers import ZeroPadding1D,Input, Convolution1D, UpSampling1D, AveragePooling1D, MaxPooling1D
from keras.layers import Input
from keras.models import Sequential
from kModel import kModel
from scipy.optimize import fmin_l_bfgs_b
from scipy.io import wavfile
import pdb

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = evaluation(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


def start(model, sRate, cSignal, sSignal):
    sampleRate=sRate
    content_w=0.025
    print 'sRate ', sRate
    print 'cSignal ', cSignal.shape
    print 'sSignal', sSignal.shape
    global countSamples
    countSamples=cSignal.shape[1]
    style_w=1.0
    contentSignal=K.variable(shapeArray(cSignal))
    styleSignal=K.variable(shapeArray(sSignal))
    #placeholder=K.placeholder(np.random.random((1,countSamples,1)))
    placeholder=K.placeholder((1,cSignal.shape[1],1))
    pdb.set_trace()
    inputTensor=K.concatenate([contentSignal,styleSignal,placeholder],axis=0)
    kModel=model# this is the object, not the model
    input_au=Input(shape=(3,None,1))#den eimai sigouros
    print '1__'

#    build()
    kModel.buildAutoEncoder(False,input_au)
    netModel,outputLayers=kModel.getModel()
    pdb.set_trace()
    output={}#dict([(layer.name, layer.output) for layer in netModel.layers])
    # output['conv1']=kModel.get_activations1(inputTensor)
    # output['conv2'] = kModel.get_activations2(inputTensor)
    # output['conv3'] = kModel.get_activations3(inputTensor)
    # output['conv4'] = kModel.get_activations4(inputTensor)
    # output['conv5'] = kModel.get_activations5(inputTensor)

    # for key in outputLayers:
    #     outputLayers[key].predict_on_batch()

    print '2__'
    loss=K.variable(0.0)
    pdb.set_trace() #check output shape
    fMap=outputLayers['encoder1'].predict_on_batch(cSignal)

    contentFMap=fMap[0,:,:]
    placeholderFMap=fMap[2,:,:]
    loss+=content_w*contentLoss(contentFMap,placeholderFMap)

    fLayers=['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    for l in fLayers:
        fMap=output[l]
        styleFMap=fMap[1,:,:]
        placeholderFMap=fMap[2,:,:]
        styleL=styleLoss(styleFMap,placeholderFMap)
        loss+=(style_w/len(fLayers))*styleL

    gradients=K.gradients(loss,placeholder)
    outGradients=[loss]
#        den 3erw ti paizei edw opote paei comment
    if type(gradients) in {list, tuple}:
        outGradients += gradients
    else:
        outGradients.append(gradients)

    global f_outputs
    f_outputs= K.function([placeholder], outGradients)
    evaluator = Evaluator()
    x = shapeArray(np.random.random((1,countSamples,1)))
    for i in range(10):
        print('Start of iteration', i)
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        wavfile.write('outFiles/output%d.wav' %i, sampleRate,x)

def shapeArray(ar):
    return ar

def evaluation(x):
    x = x.reshape((1,countSamples,1))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

def getGram(matrix):
    assert K.ndim(matrix) == 2 , "gram ndim not 2"
    gram=K.dot(matrix.T,matrix)
    return gram
#        features = K.batch_flatten(x)
#        gram = K.dot(features, K.transpose(features))

def styleLoss(style,placehold):
    assert K.ndim(style)==2 ,"style ndim not 2"
    assert K.ndim(placehold==2), "placeholder ndim not 2"
#        predStyle=netModel.predict(styleSignal) #feature maps
    Sg=getGram(style)
    Pg=getGram(placehold)
    return K.sum(K.square(Sg-Pg))/ K.sum(K.square(Sg)) #may be Sg-Pg

def contentLoss(content,placehold):
    Pg=getGram(placehold)
    Cg=getGram(content)
    loss=K.sum(K.square(Pg-Cg))
    return loss
