import pickle


def dump(nn, model):
    '''Dumps the params of a given NeuralNetwork into a dictionary.
    
    The dictionary must have the attribute name of the layer as a key.
    E.g.: nn.layer1 --> model['layer1']
    
    Agrs:
        nn: The NeuralNetwork
        model: A dict where the params get stored at
    '''
    for k,v in model.items():
        layer = getattr(nn, k)
        model[k] = (layer.w.copy(), 0 if layer.b is 0 else layer.b.copy())
    pass


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
        model: A dict with the parameters.
        fname: The path/file name to be stored at.
    '''
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
    dump(nn, nn.model)
    pass
