import numpy as np 
from scipy.misc import imread, imresize

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x /255.0
    return x 