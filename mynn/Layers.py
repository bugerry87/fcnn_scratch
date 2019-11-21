import numpy as np

class Layer:
    '''A typedef for Neural Network Layers'''
    pass

class FcLayer(Layer):
    '''A simple Fully-Connected Layer
    
    Instead of "feed forward" this impl "pull_forward"
    the output from the lower layer as input to the upper layer
    and performs the back-propagation via "push_backward".
    For "pull_forward" an input generator must be assigned to this layer.
    "pull_forward" is such an input generator,
    such that layers can be connected with each other,
    pulling the output of the underlaying layer to the upper layers,
    up to the final output-layer.
    The back-propagation is a simple function-pointer,
    to the "push_backward" function of the underlaying layer
    '''
    
    def __init__(self,
                 params,
                 act_func,
                 input_func,
                 backprop,
                 clipping=False
                ):
        '''Initializer of FcLayer
        
        Args:
            params: A tuble that contains weights and bias.
            act_func: The activation function.
            input_func: A function that creates an generator.
            backprop: The "push_backward" of another Layer.
            clipping: For clipping the output to a min-&-max value.
                        this may prevent number overflows.
        '''
        self.w = params[0]
        self.b = params[1]
        self.act = act_func
        self.input = input_func
        self.backprop = backprop
        self.clipping = clipping
        pass
    
    def pull_forward(self):
        '''Pulls the output from the underlaying layer. (Feed Forward)
        
        Creates a generator that pulls the input,
        performs the activation and yields the result.
        Stores a referenc to the input as "x"
        and the result as "y".
        If the "act_func" is not monotonic,
        an intermediet result will be stored in "z".
        
        Yields:
            The activation of (wx + b)
        '''
        def z():
            return np.dot(self.x, self.w) + self.b
        
        for self.x in self.input():
            if self.act.monotonic():
                #safe the storage
                self.y = self.act.f(z())
            else:
                self.z = z()
                self.y = self.act.f(self.z)
            yield self.y
    
    def push_backward(self, dZ, lr):
        '''Pushes the loss to the underlaying layer. (Backpropagation)
        
        Performs the beck-propagation recursively based on the given gradient "dZ".
        
        Args:
            dZ: The gradient computed by NeuralNetwork.train
            lr: The current learning-rate
        '''
        dZ *= self.act.d(self.y if self.act.monotonic else self.z) #dz+1/dy * dy/dz
        dw = np.dot(self.x.T, dZ) #zd/wd
        if self.backprop:
            self.backprop(np.dot(dZ, self.w.T), lr) #dz/dy-1 = W * dz
        self.w += dw * lr if not self.clipping else np.clip(dw * lr, -self.clipping, self.clipping)
        if not self.b is 0:
            self.b += np.sum(dZ, axis=0) * lr
        pass


def init_FC_params(inp, outp, scale=1, bias=True, init_func=np.random.randn):
    '''Initialize parameters for FcLayers.
    
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