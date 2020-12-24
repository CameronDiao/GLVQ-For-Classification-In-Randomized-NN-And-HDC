import numpy as np
import pandas as pd

from models.ConvRVFL.base import ConvRVFL
from classifiers import rlms

class ConvRVFLUsingRLMS(ConvRVFL):
    def __init__(self, train_set, n, lmb):
        super().__init__(train_set, n)
        self.lmb = lmb

    def readout(self, inputs, labels):
        return rlms(inputs, labels, self.lmb)

    def train(self):
        train_features = self.preprocess(self.train_set)
        act_matrix = self.activations(train_features)
        train_labels = pd.get_dummies(self.train_set["clase"]).values
        w = self.readout(act_matrix, train_labels)
        return w

    def score(self, test_set):
        w = self.train()
        test_features = self.preprocess(test_set)
        act_matrix = self.activations(test_features)
        test_pred = act_matrix @ w
        test_pred = np.argmax(test_pred, axis=1)

        test_labels = test_set["clase"].values
        num_correct = np.sum(test_pred == test_labels)
        return num_correct / len(test_labels)
