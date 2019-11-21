import numpy as np

class FeedFunc():
    '''typedef for feed forward functions'''
    pass

class InnerProduct(FeedFunc):
    def f(x, w):
        return np.dot(x, w)
    
    def dw(x, dZ):
        return np.dot(x.T, dZ)
    
    def dz(dZ, w):
        return np.dot(dZ, w.T)