import numpy as np
from scipy.special import xlogy, xlog1py


class LossFunc:
    '''A typedef for Loss Functions'''
    pass


class SOS(LossFunc):
    '''Sum of Squares error'''
    def f(t, y):
        return np.sum(np.power(t - y, 2))
    
    def d(t, y):
        return (t - y) * 2


class RMS(LossFunc):
    '''Root Mean Square error'''
    def f(t, y):
        return np.power(SOS.f(t, y) / t.size, 0.5)
    
    def d(t, y):
        return (t - y) / t.size


class CE(LossFunc):
    '''Cross Entropy'''
    def f(t,y):
        #return -np.sum(np.sum(np.dot(t, np.log(y.T)), axis=1)) / t.shape[1] #multy class
        #return -np.sum(np.dot(t, np.log(y.T)) + np.dot(1 - t, np.log(1 - y.T))) / t.shape[1]
        return -np.sum(xlogy(t, y) + xlog1py(1 - t, -y)) / t.shape[1]
    
    def d(t,y):
        return (t - y) / t.shape[1]


class AMax(LossFunc):
    '''Arg Max Pooling'''
    def f(t,y):
        return np.sum(np.argmax(t, axis=1) != np.argmax(y, axis=1)) / t.shape[0]
    
    def d(t,y):
        z = np.zeros(y.shape)
        z[:,np.argmax(y, axis=1)[0]] = 1
        return (t - z) / t.shape[1]