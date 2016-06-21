from keras import backend as K
import numpy as np
from keras.layers import ZeroPadding1D,Input, Convolution1D, UpSampling1D, AveragePooling1D, MaxPooling1D
from keras.layers import Input
from scipy.optimize import fmin_l_bfgs_b
from scipy.io import wavfile
import theano as T
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

global gradient_function
global loss_function
def start(model, sRate, cSignal, sSignal):
    sampleRate=sRate
    content_w=0.025
    print 'sRate ', sRate
    print 'cSignal ', cSignal.shape
    print 'sSignal ',sSignal[0].shape
    global countSamples
    countSamples=cSignal.shape[1]
    noise=np.random.random((1,countSamples,1))
    style_w=1.0
    #contentSignal=K.variable(shapeArray(cSignal))
   # styleSignal=K.variable(shapeArray(sSignal))
    #placeholder=K.placeholder(np.random.random((1,countSamples,1)))
    #placeholder=K.placeholder((1,cSignal.shape[1],1))
    #inputTensor=K.concatenate([contentSignal,styleSignal,placeholder],axis=0)
    kModel=model# this is the object, not the model
    input_au=Input(shape=(3,None,1))#den eimai sigouros
    print '1__'

#    build()
    kModel.buildAutoEncoder(False,input_au)
    netModel,outputLayers=kModel.getModel()
    output={}#dict(c[(layer.name, layer.output) for layer in netModel.layers])
    # output['conv1']=kModel.get_activations1(inputTensor)
    # output['conv2'] = kModel.get_activations2(inputTensor)
    # output['conv3'] = kModel.get_activations3(inputTensor)
    # output['conv4'] = kModel.get_activations4(inputTensor)
    # output['conv5'] = kModel.get_activations5(inputTensor)

    # for key in outputLayers:
    #     outputLayers[key].predict_on_batch()

    print '2__'
    # loss=K.variable(0.0)
    # pdb.set_trace() #check output shape
    # fMap=outputLayers['encoder1'].predict_on_batch(cSignal)

    # contentFMap=fMap[0,:,:]
    # placeholderFMap=fMap[2,:,:]
    # loss+=content_w*contentLoss(contentFMap,placeholderFMap)

    X=Input(shape=(countSamples,1))
    loss=0
    c=[]
    g=[]
    s=[]
    count=0
    for l in outputLayers:
        print count,' ', cSignal.shape, ' ',str(l)
        print 'in shape', outputLayers[l].get_input_shape_at(0)
        print 'out shape', outputLayers[l].get_output_shape_at(0)
        tempC=outputLayers[l].predict(cSignal)
        tempG=outputLayers[l](X)
        c=tempC[0]
        g=tempG[0]#tensor
        sGram=1
        for style in sSignal:
            tempS=outputLayers[l].predict(style)
            s=tempS[0]
            sGram*=getGram(s)/style.shape[1]
        sGram**=1.0/len(sSignal) #sSignal sould be a list
        loss+=style_w * styleLoss(sGram,g)
        count+=1
    gradient_function=T.function([X], K.flatten(K.gradients(loss,X)) ,allow_input_downcast=True)
    loss_function=T.function([X],loss,allow_input_downcast=True)

    bounds = [[-0.9, 0.9]]
    bounds = np.repeat(bounds, countSamples, axis=0)

    print("optimizing")
    pdb.set_trace()
    y, Vn, info = fmin_l_bfgs_b(
        evaluation,
        noise.astype(np.float64).flatten(),
        bounds=bounds,
        factr=0.0, pgtol=0.0,
        maxfun=30000,  # Limit number of calls to evaluate().
        iprint=1,
        approx_grad=False,
        callback=optimization_callback)

    wavfile.write('output/output.wav', countSamples, y.astype(np.int16))
    print("done.")

def optimization_callback(xk):
    global iteration_count
    if iteration_count % 10 == 0:
        current_x = np.copy(xk)
        wavfile.write('output%d.wav' % iteration_count, countSamples, current_x.astype(np.int16))
    iteration_count += 1

def shapeArray(ar):
    return ar

def evaluation(x):
    tempX=np.reshape(x, (1, countSamples, 1)).astype(np.float32)
    gradients = gradient_function(tempX)
    total_loss=loss_function(tempX)
    return total_loss.astype(np.float64),gradients.astype(np.float64)


def getGram(matrix):
#    assert K.ndim(matrix) == 2 , "gram ndim not 2"
    gram=K.dot(matrix.T,matrix)
    return gram
#        features = K.batch_flatten(x)
#        gram = K.dot(features, K.transpose(features))

def styleLoss(style,placehold):
 #   assert K.ndim(style)==2 ,"style ndim not 2"
  #  assert K.ndim(placehold==2), "placeholder ndim not 2"
#        predStyle=netModel.predict(styleSignal) #feature maps
    Sg=style
    Pg=getGram(placehold)
    return K.sum(K.square(Sg-Pg))/ K.sum(K.square(Sg)) #may be Sg-Pg

def contentLoss(content,placehold):
    Pg=getGram(placehold)
    Cg=getGram(content)
    loss=K.sum(K.square(Pg-Cg))
    return loss
