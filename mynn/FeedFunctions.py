import numpy as np

class FeedFunc():
    '''typedef for feed forward functions'''
    pass

class InnerProduct(FeedFunc):
    def f(x, w):
        return np.dot(x, w)
    
    def dw(x, dZ):
        return np.dot(x.T, dZ)
    
    def dy(dZ, w):
        return np.dot(dZ, w.T)
    
    def db(dZ):
        return dZ.sum(axis=0)
    
    def init_params(inp, outp, scale=1, bias=True, init_func=np.random.randn):
        '''Initialize parameters.
        
        Args:
            inp: How many inputs the layer should absorb.
            outp: How many outputs the layer should produce.
            scale: Scaling of the initial parameters.
            bias: True, if a bias should be used, otherwise False.
            init_func: Alter the initialization function.
                (default) np.random.randn
        Returns:
            weights: A 2-dimensional matrix of weights.
            bias: Either 0 or a vector of biases for each node.
        '''
        weights = init_func(inp, outp)
        while not np.all(weights):
            malicious = weights==0.0
            weights[malicious] = init_func(malicious.shape)
        weights *= scale
        return weights, np.zeros(outp) if bias else 0