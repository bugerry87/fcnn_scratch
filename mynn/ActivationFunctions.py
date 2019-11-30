import numpy as np


class ActFunc:
    '''A typedef for Activation Functions'''
    def monotonic(self=None):
        '''Activation functions are usually monotonic.
        
        Otherwise, override this function with False.
        '''
        return True
    
    def f(x):
        '''The original equition'''
        pass
    
    def d(y):
        '''The derivative'''
        pass


class Sigmoid(ActFunc):
    def __init__(self, relax):
        self.r = relax

    def f(self, z):
        return 1 / (1 + np.exp(-z / self.r))
    
    def d(self, S):
        '''Derivative of Sigmoid
        
        Sigmoid is monotonic,
        such that we can forward the activation potential directly.
        
        Args:
            S = Sigmoid.f(z)
            
        Returns:
            Derivative of Sigmoid.f(z)
        '''
        return S * (1 - S)


class TanH(ActFunc):
    def f(x):
        return np.tanh(x)
    
    def d(a):
        '''Derivative of TanH
        
        TanH is monotonic,
        such that we can forward the activation potential directly.
        
        Args:
            a = TanH.f(x)
            
        Returns:
            Derivative of TanH.f(x)
        '''
        return 1 - np.power(a, 2)


class ReLU(ActFunc):
    def f(x):
        return np.maximum(x,0).astype(float)
    
    def d(y):
        '''Derivative of ReLU
        
        ReLU is monotonic,
        such that we can forward the activation potential directly.
        '''
        return (y > 0).astype(float)


class Identity(ActFunc):
    def f(x):
        return x
    
    def d(y):
        return np.ones(y.shape)


