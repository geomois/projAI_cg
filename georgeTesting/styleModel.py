from keras import backend as K
import numpy as np
from keras.layers import ZeroPadding1D,Input, Convolution1D, UpSampling1D, AveragePooling1D, MaxPooling1D
from keras.layers import Input
from scipy.optimize import fmin_l_bfgs_b
from scipy.io import wavfile
import theano as T
from normalize import *
import soundfile as sf
import pdb

def start(model, sRate, cSignal, sSignal):
    sampleRate=sRate
    content_w=0.025
    global normRate
    cSignal,normRate=normalize(shapeArray(cSignal))
    print 'sRate ', sRate
    print 'cSignal ', cSignal.shape
    print 'sSignal ',sSignal[0].shape
    global countSamples
    countSamples=cSignal.shape[1]
    noise=np.random.random((1,countSamples,1))
    style_w=1.0
    kModel=model
    input_au=Input(shape=(3,None,1))
#####build
    kModel.buildAutoEncoder(False,input_au)
    netModel,outputLayers=kModel.getModel()

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
        g=tempG[0]#K tensor
        sGram=1
        for style in sSignal:
            tempS=outputLayers[l].predict(style)
            s=tempS[0]
            sGram*=getGram(s)/style.shape[1]
        sGram**=1.0/len(sSignal) #sSignal sould be a list

        loss+=style_w * styleLoss(sGram,g) +content_w * contentLoss(c,g)
        count+=1

    loss *= 10e7
    global gradient_function
    global loss_function
    gradient_function=T.function([X], T.flatten(T.grad(loss,X)) ,allow_input_downcast=True)
    loss_function=T.function([X],loss,allow_input_downcast=True)
    global iteration_count
    iteration_count = 0

    bounds = [[-1.0, 1.0]]
    bounds = np.repeat(bounds, countSamples, axis=0)

    print("optimizing")
    pdb.set_trace()
    opt, Vn, info = fmin_l_bfgs_b(
        evaluation,noise.astype(np.float64).flatten(),bounds=bounds,factr=0.0, pgtol=0.0,maxfun=30000,  # Limit number of calls to evaluate().
        iprint=1,approx_grad=False,callback=optimization_callback)

    sf.write('../outFiles/output.wav', opt.astype(np.float32), countSamples)
    print("done.")

def optimization_callback(xk):
    global iteration_count
    if iteration_count % 10 == 0:
        current_x = np.copy(xk)
        print current_x.shape
        current_x=denormalize(current_x,normRate)
        wavfile.write('../outFiles/iter/output%d.wav' % iteration_count, countSamples, current_x.astype(np.int16))
    iteration_count += 1

def shapeArray(ar):
    depth=ar.shape[0]
    x=ar.shape[1]
    return ar.resahpe((1,depth*x,1))

def evaluation(x):
    tempX=np.reshape(x, (1, countSamples, 1)).astype(np.float32)
    gradients = gradient_function(tempX)
    total_loss=loss_function(tempX)
    return total_loss.astype(np.float64),gradients.astype(np.float64)


def getGram(matrix):
#    assert K.ndim(matrix) == 3 , "gram ndim not 3"
    gram=K.dot(matrix.T,matrix)
    return gram
#        features = K.batch_flatten(x)
#        gram = K.dot(features, K.transpose(features))

def styleLoss(style,placehold):
 #   assert K.ndim(style)==2 ,"style ndim not 2"
  #  assert K.ndim(placehold==2), "placeholder ndim not 2"
#   predStyle=netModel.predict(styleSignal) #feature maps
    Sg=style
    Pg=getGram(placehold)
    return K.sum(K.square(Sg-Pg))/ T.sum(T.square(Sg))

def contentLoss(content,placehold):
    Pg=getGram(placehold)
    Cg=getGram(content)
    loss=K.sum(K.square(Pg-Cg))
    return loss
