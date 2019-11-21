import numpy as np

class NeuralNetwork:
    def __init__(self, train_set, val_set, cost_func, loss_func):
        '''Initialization of NeuralNetwork
        
        Args:
            train_set: A generator that iterates through training data.
            val_set: A generator that iterates through validation data.
            cost_func: A cost function minimized during training.
            loss_func: A loss function for performance validation.
        '''
        self.train_set = train_set
        self.val_set = val_set
        self.cost = cost_func
        self.loss = loss_func
        self.training = False
        pass
    
    def train(self, output_layer, lr, lr_dec):
        '''A generator function processes and iterates through the training steps.
        
        Computes the cost, performs the Back Propagation and updates the params.
        Yields for each batch.
        
        Args:
            output_layer: The output layer whose Y gets tested against the cost function.
            lr: The learning-rate, determines how much of the gradient gets adapted to the params.
            lr_dec: A decay rate, decays the learning rate per epoch.
        
        Yields:
            L: The cost
        '''
        self.lr = lr
        self.lr_dec = lr_dec
        self.training = True
        for Y in output_layer.pull_forward():
            L = np.squeeze(self.cost.f(self.T, Y))
            dL = self.cost.d(self.T, Y)
            output_layer.push_backward(dL, self.lr)
            self.lr -= self.lr * self.lr_dec
            yield L
        pass
    
    def val(self, output_layer):
        '''A generator function processes and iterates through the validation steps.
        
        Evaluates the loss.
        Yields for each batch.
        
        Args:
            output_layer: The output layer whose Y gets tested against the loss function.
        
        Yields:
            L: The loss
        '''
        self.training = False
        for Y in output_layer.pull_forward():
            L = self.loss.f(self.T, Y)
            yield L
        pass
    
    def gen_input(self):
        '''A generator that pulls the inputs either from the training or the validation set.
        
        Switch between the sets by setting the training flag.
        
        Yields:
            X: A sample either from train_set or val_set.
        '''
        if self.training:
            for self.X, self.T, self.epoch, self.step in self.train_set():
                yield self.X
        else:
            for self.X, self.T, _, _ in self.val_set():
                yield self.X
        pass