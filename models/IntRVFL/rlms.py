import numpy as np
import pandas as pd

from .base import IntRVFL
from classifiers import rlms

class IntRVFLUsingRLMS(IntRVFL):
    def __init__(self, train_set, n, kappa, lmb):
        super().__init__(train_set, n, kappa)
        self.lmb = lmb

    def readout(self, inputs, labels):
        return rlms(inputs, labels, self.lmb)

    def train(self):
        train_features = self.preprocess(self.train_set)
        enc_matrix = self.encodings(train_features)
        act_matrix = self.activations(enc_matrix)
        train_labels = pd.get_dummies(self.train_set["clase"]).values
        w = self.readout(act_matrix, train_labels)
        return w

    def score(self, test_set):
        w = self.train()
        test_features = self.preprocess(test_set)
        enc_matrix = self.encodings(test_features)
        act_matrix = self.activations(enc_matrix)
        test_pred = act_matrix @ w
        test_pred = np.argmax(test_pred, axis=1)

        test_labels = test_set["clase"].values
        num_correct = np.sum(test_pred == test_labels)
        return num_correct / len(test_labels)
