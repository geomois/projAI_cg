from __future__ import print_function
from scipy.misc import imread, imresize, imsave
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import os
import argparse
import h5py
import pickle
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras import backend as K
from keras.datasets import mnist
import numpy as np
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model

(x_train, _), (x_test, _) = mnist.load_data()

total_variation_weight = 1.
style_weight = 1.
content_weight = 0.025

img_width = 28
img_height = 28

def makeNet(inshape):
        layersdict=dict()
        input_img = Input(shape=inshape)
        x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
        layersdict['conv_1']=Model(input=input_img,output=x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
        layersdict['conv_2']=Model(input=input_img,output=x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
        layersdict['conv_3']=Model(input=input_img,output=x)
        encoded = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
        layersdict['conv_4']=Model(input=input_img,output=x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
        layersdict['conv_5']=Model(input=input_img,output=x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(16, 3, 3, activation='relu')(x)
        layersdict['conv_6']=Model(input=input_img,output=x)
        x = UpSampling2D((2, 2))(x)
        decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)
        layersdict['conv_7']=Model(input=input_img,output=decoded)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        pickle.dump( layersdict, open( "layers.p", "wb" ) )
        return autoencoder, layersdict

def trainNet(autoencoder,x_train,x_test):
        autoencoder.fit(x_train, x_train,
                        nb_epoch=25,
                        batch_size=128,
                        shuffle=True,
                        validation_data=(x_test, x_test),
                        callbacks=[])
        autoencoder.save_weights('keras_w_mnist',overwrite=True)

def loadNet(autoencoder):
        assert os.path.exists('keras_w_mnist'), 'Model weights file not found.'
        autoencoder.load_weights('keras_w_mnist')

net,layersdict = makeNet((1,28,28))
#trainNet(net,x_train,x_test)
loadNet(net)

def gram_matrix(x):
    features = K.batch_flatten(x)
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination): #inputs are tensors
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 1 #shouldn't it be the number of filters instead of channels?? 
    size = img_width * img_height
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))
    #hmmm....i think is wrong having the flattened version and giving scalars and then subtracting them....

#x=np.resize(base,(1,28,28))
#y=np.resize(combination,(1,28,28))
#t=K.variable(x)
#f=K.variable(y)
#res=style_loss(t,f)
#print("res",res.eval())
#print("ginetai kai xoris na metatrepso to input se tensor! ;) ")
#x=np.resize(base,(1,28,28))
#y=np.resize(combination,(1,28,28))
#res=style_loss(x,y)
#print("res ",res.eval()) #to apotelesma ine tensor!! 

def content_loss(base, combination):
    return K.sum(K.square(combination - base))

base=x_train[2]
style=x_train[4]
base=np.resize(base,(1,28,28))
style=np.resize(style,(1,28,28))

base_image = K.variable(base)
style_image = K.variable(style)
combination=Input(shape=(1,28,28))
#combination_image = K.placeholder((1, 1, img_width, img_height))
#input_tensor = K.concatenate([base_image,style_reference_image,combination_image], axis=0)
input_tensor=np.vstack( (base ,style , combination) )

outputs_dict_base=dict()
outputs_dict_noise=dict()
outputs_dict_combination=dict()

for l_name in layersdict:
        outputs_dict_base[l_name]=layersdict[l_name].predict(base)
        outputs_dict_noise[l_name]=layersdict[l_name].predict(noise)
        outputs_dict_combination[l_name]=layersdict[l_name](combination)

loss=K.variable(0.)

"""
layer_features=outputs_dict['convolution2d_4']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(base_image_features,combination_features)

style_layers = ['convolution2d_3', 'convolution2d_4', 'convolution2d_5']

for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl

grads = K.gradients(loss, combination_image)

outputs = [loss]
if type(grads) in {list, tuple}:
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)
def eval_loss_and_grads(x):
    x = x.reshape((1, 1, img_width, img_height))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

x = np.random.uniform(0, 255, (1, 1, img_width, img_height))
for i in range(3):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(loss, x.flatten(), fprime=grads, maxfun=20)
    print('Current loss value:', min_val)
    # save current generated image
    img = deprocess_image(x.reshape((1, img_width, img_height)))
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
print('Iteration %d completed in %ds' % (i, end_time - start_time))

"""
