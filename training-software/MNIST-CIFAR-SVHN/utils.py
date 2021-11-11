import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Convolution2D, Activation, Flatten, MaxPooling2D,Input,Dropout,GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.framework import ops
#from multi_gpu import make_parallel

def binarize(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    
    Modifications by Lev
    '''
#     clipped = K.clip(x,-1,1)
    clipped = tf.cast(K.clip(x,-1,1),tf.float32)
    rounded = K.sign(clipped+tf.constant(1e-17))  #Nudges from zeros, need to find better solution for later
#     rounded = tf.cast(roundeds)
    #rounded = K.sign(clipped) #Does not work when input is zero, outputs zero
    return clipped + K.stop_gradient(rounded - clipped)