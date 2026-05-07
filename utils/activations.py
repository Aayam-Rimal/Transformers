import numpy as np

def softmax(x):

    x= x- np.max(x, axis=-1, keepdims=True )
    exp_x= np.exp(x)
    return exp_x/np.sum(exp, axis=-1, keepdims=True)


def relu(x):
    return np.maximum(0,x)


def sigmoid(x):
    return 1/(1+np.exp(-x))

