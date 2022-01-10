import numpy as np

def softmax(x, t=1):
    """"
    Applies the softmax temperature on the input x, using the temperature t
    """
    X = x - np.amax(x)
    return np.exp(X/t) / np.sum(np.exp(X/t))