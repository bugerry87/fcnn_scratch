import numpy as np
import matplotlib.pylab as plt
from .Utilities import *


def train_epoch(nn, lr, lr_dec):
    '''Trains a NeuralNetword without any visualizations.
    
    Args:
        nn: The Neural Network to be trained.
        lr: The learning-rate applied onto the gradient update.
        lr_dec: Decay of the learning-rate per epoch.
    '''
    cost = 0
    loss = 0
    
    for Z in nn.train(nn.Out, lr, lr_dec):
        cost += Z
        pass
    
    for Zv in nn.val(nn.Out):
        loss += Zv
        pass
                
    return loss, cost


def train_n_plot(nn, epochs, lr, lr_dec, checkpoint=None, every=None):
    '''Trains a NeuralNetwork and plots the learning-curves.
    
    Args:
        nn: The Neural Network to be trained.
        epochs: Number of epochs. How many times to iterate the entire dataset.
        lr: The learning-rate applied onto the gradient update.
        lr_dec: Decay of the learning-rate per epoch.
        checkpoint: A file/path name prefix where the checkpoints to be stored at.
            (default) None = No checkpoints will be stored.
        every: Store a checkpoint for every X epoch.
            Requires the parameter 'checkpoint'
            (default) None = Only the best result will be stored.
    
    Returns:
        best_epoch: The number of the epoch with the best loss.
        best_loss: The loss-value (score) of the best model.
    '''
    training = []
    validation = []
    best_loss = None
    best_epoch = 0
    past = nn.step
    
    if not hasattr(nn, 'model'):
        nn.model = {}
    
    for epoch in range(epochs):
        loss, cost = train_epoch(nn, lr, lr_dec)
        N = nn.step - past
        training.append(cost/N)
        validation.append(loss/N)
        past = nn.step
        if best_loss is None or loss <= best_loss:
            best_loss = loss
            best_epoch = epoch
            dump(nn, nn.model)
        if checkpoint:
            save(nn.model, '{}_best.pkl'.format(checkpoint))
        if checkpoint and every and not epoch % every:
            dump(nn, nn.model)
            save(nn.model, '{}_{}.pkl'.format(checkpoint, epoch))
    
    ax_l = plt.figure().subplots(1)
    ax_l.plot(range(epochs), training, label='train')
    ax_l.set_xlabel("Epoch")
    ax_l.set_ylabel("Cost")

    ax_r = ax_l.twinx()
    ax_r.plot(range(epochs), validation, 'g:', label='val')
    ax_r.scatter(best_epoch, best_loss, color='g', marker='D', zorder=3, label='best')
    ax_r.set_ylabel("Loss")

    ax_l.legend(loc=2)
    ax_r.legend()

    plt.title("Learning curves")
    plt.show()
    print("Best Epoch:", best_epoch, "with a Loss of:", best_loss)
    return best_epoch, best_loss
