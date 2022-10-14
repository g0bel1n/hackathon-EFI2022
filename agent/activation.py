import numpy as np


tanh = np.tanh

sigmoid = lambda x: 1 / (1 + np.exp(-x))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

softmax_o_sigmoid = lambda x: softmax(sigmoid(x))