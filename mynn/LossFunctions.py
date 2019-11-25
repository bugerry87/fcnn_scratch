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
        return np.power(SOS.f(t, y) / t.shape[0], 0.5)
    
    def d(t, y):
        return (t - y)


class CrossEntropy(LossFunc):
    '''Cross Entropy'''
    def f(t,y):
        #return -np.sum(np.dot(t, np.log(y.T)) + np.dot(1 - t, np.log(1 - y.T))) / t.shape[0]
        #return -np.sum(xlogy(t, y) + xlog1py(1 - t, -y)) / t.shape[0]
        return -np.sum(np.xlog1py(t, y)) / t.shape[0] #multy class
    
    def d(t,y):
        return (t - y)


class ArgMaxPooling(LossFunc):
    '''Arg Max Pooling'''
    def f(t,y):
        return np.sum(np.argmax(t, axis=1) != np.argmax(y, axis=1)) / t.shape[0]
    
    def d(t,y):
        z = np.zeros(y.shape)
        z[:,np.argmax(y, axis=1)[0]] = 1
        return (t - z) / t.shape[1]