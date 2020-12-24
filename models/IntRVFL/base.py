import numpy as np
from numba import njit, vectorize, float64

@vectorize([float64(float64, float64)])
def quantize(elem, n):
    return int(round(elem * n))

@njit
def encoding_helper(dataset, n):
    enc_matrix = np.zeros((dataset.shape[0], n, dataset.shape[1]))
    for row in range(dataset.shape[0]):
        x = quantize(dataset[row], np.float64(n))
        f = np.ones((x.size, n))
        for vec in range(f.shape[0]):
            f[vec][0:int(x[vec])] = -1
        enc_matrix[row] = f.T
    return enc_matrix

@njit
def activation_helper(dataset, w_in, kappa):
    act_matrix = np.zeros((dataset.shape[0], dataset.shape[1]))
    for row in range(dataset.shape[0]):
        x = dataset[row]
        h = np.sum(np.multiply(x, w_in), axis=1)
        for i, elem in np.ndenumerate(h):
            if elem >= kappa:
                h[i] = kappa
            elif elem <= -kappa:
                h[i] = -kappa
        act_matrix[row] = h
    return act_matrix

class IntRVFL:
    def __init__(self, train_set, n, kappa):
        self.train_set = train_set
        self.n = n
        self.kappa = kappa
        self.k = len(train_set.columns) - 1
        self.w_in = np.random.choice([-1, 1], size=(self.n, self.k))

    def preprocess(self, dataset):
        return dataset.drop(["clase"], axis=1).values

    def encodings(self, dataset):
        return encoding_helper(dataset, self.n)

    def activations(self, dataset):
        return activation_helper(dataset, self.w_in, self.kappa)

    def readout(self, inputs, labels):
        pass

    def train(self):
        pass

    def score(self, test_set):
        pass