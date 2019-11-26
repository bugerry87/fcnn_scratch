import pickle
from .Layer import Layer
from .NeuralNetwork import NeuralNetwork

def dump(nn, model=None):
    '''Dumps the params of a given NeuralNetwork into a dictionary.
    
    E.g.: nn.layer1 --> model['layer1'] = nn.layer1
    
    Agrs:
        nn: The NeuralNetwork
        model: Optional output parameter
    '''
    if not model:
        model = {}
    
    for k,v in nn.__dict__.items():
        if isinstance(v, Layer):
            model[k] = (v.w.copy(), 0 if v.b is 0 else v.b.copy())
    return model


def assign(model, nn):
    '''Assigns the params of a given dictionary into a NeuralNetwork.
    
    The dictionary must have the attribute name of the layer as a key.
    E.g.: model['layer1'] --> nn.layer1
    
    Agrs:
        model: A dict with parameters
        nn: The NeuralNetwork where the params get stored at
    '''
    for k,v in model.items():
        layer = getattr(nn, k)
        layer.w = model[k][0].copy()
        layer.b = 0 if model[k][1] is 0 else model[k][1].copy() 
    pass


def save(model, fname):
    '''Saves the model to a pickle file.
    
    The model is a dictionary.
    The weights and bias should be stored layer-wise as a tuble.
    Note: It's recommended to coincide the naming!
    E.g.: model['layer1'] = (nn.layer1.w, nn.layer1.b)
    
    Args:
        model: A NeuralNetwork or dict with the parameters.
        fname: The path/file name to be stored at.
    '''
    if isinstance(model, NeuralNetwork):
        model = dump(model)
    
    with open(fname, 'wb') as file:
        pickle.dump(model, file)
    pass


def load(nn, fname):
    '''Loads a pickle file to a NeuralNetwork.
    
    Note: Make sure that the layers have the same naming as in the file.
    
    Args:
        nn: The NeuralNetwork
        fname: The path/file name to be loaded.
    '''
    with open(fname, 'rb') as file:
        obj = pickle.load(file)
    for k,v in obj.items():
        if hasattr(nn, k):
            layer = getattr(nn, k)
            layer.w = v[0]
            layer.b = v[1]
    pass
