import numpy as np
import scipy as sc

def rlms(inputs, labels, lmb):
    inputs = inputs.astype(dtype=np.float32)
    labels = labels.astype(dtype=np.float32)
    id = np.diag(np.var(inputs, axis=0))
    #w = np.linalg.lstsq(inputs.T @ inputs + id * lmb, inputs.T @ labels)[0]
    w = sc.linalg.pinv(inputs.T @ inputs + id * lmb) @ inputs.T @ labels
    return w