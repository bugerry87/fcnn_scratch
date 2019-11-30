import numpy as np
from scipy.special import xlogy, xlog1py


class LossFunc:
    '''A typedef for Loss Functions'''
    pass


class SumOfSquaresError(LossFunc):
    '''Sum of Squares error'''
    def f(t, y):
        return np.sum(np.power(t - y, 2))
    
    def d(t, y):
        return (t - y) * 2


class RootMeanSquareError(LossFunc):
    '''Root Mean Square error'''
    def f(t, y):
        return np.power(SOS.f(t, y) / t.shape[-1], 0.5)
    
    def d(t, y):
        return SOS.d(t, y) / t.shape[-1]


class CrossEntropy(LossFunc):
    '''Cross Entropy'''
    def f(t,y):
        return -np.sum(xlogy(t, y) + xlog1py(1 - t, -y)) / t.shape[-1]
    
    def d(t,y):
        return (t - y) / t.shape[-1]


class ArgMaxPooling(LossFunc):
    '''Arg Max Pooling'''
    def f(t,y):
        return np.sum(np.argmax(t, axis=-1) != np.argmax(y, axis=-1))
    
    def d(t,y):
        z = np.zeros(y.shape)
        z[:,np.argmax(y, axis=1)[0]] = 1
        return (t - z) / t.shape[-1]