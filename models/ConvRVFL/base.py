import numpy as np

class ConvRVFL:
    def __init__(self, train_set, n):
        self.train_set = train_set
        self.n = n
        k = len(self.train_set.columns) - 1
        self.w_in = np.random.uniform(-1, 1, size=(self.n, k))
        self.b = np.random.uniform(-0.1, 0.1, size=self.n)

    def preprocess(self, dataset):
        return dataset.drop(["clase"], axis=1).values

    def activations(self, dataset):
        h_matrix = dataset @ self.w_in.T + self.b
        h_matrix = 1 / (1 + np.exp(-h_matrix))
        return h_matrix

    def readout(self, inputs, labels):
        pass

    def train(self):
        pass

    def score(self, test_set):
        pass