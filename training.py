import numpy as np
import pandas as pd

def activations(x, w_in, b, g):
    h = w_in.dot(x) + b
    func = np.vectorize(g)
    h = func(h)
    return h

def activation_matrix(dataset, w_in, b, g):
    h_matrix = []
    for feature in dataset.index:
        x = dataset.loc[feature]
        x = x.to_frame()
        h = activations(x, w_in, b, g)
        h = pd.DataFrame(data=np.transpose(h), index=[feature])
        h_matrix.append(h)
    h_matrix = pd.concat(h_matrix, axis=0)
    return h_matrix

def readout_matrix(h_matrix, y_matrix, lmb, n):
    id_matrix = np.identity(n)
    inner = (h_matrix.transpose().dot(h_matrix)).add((id_matrix * lmb))
    inverse_inner = pd.DataFrame(np.linalg.pinv(inner.values), inner.columns, inner.index)
    w_out = inverse_inner.dot(h_matrix.transpose()).dot(y_matrix)
    return w_out