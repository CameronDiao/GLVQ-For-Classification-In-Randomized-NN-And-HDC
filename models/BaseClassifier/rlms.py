import numpy as np
import pandas as pd

from models.BaseClassifier.base import BaseClassifier
from classifiers import rlms

class RLMSClassifier(BaseClassifier):
    def __init__(self, train_set, lmb):
        super().__init__(train_set)
        self.lmb = lmb

    def model(self, inputs, labels):
        return rlms(inputs, labels, self.lmb)

    def train(self):
        train_features = self.preprocess(self.train_set)
        train_labels = pd.get_dummies(self.train_set["clase"]).values
        theta = self.model(train_features, train_labels)
        return theta

    def score(self, test_set):
        theta = self.train()
        test_features = self.preprocess(test_set)
        test_pred = test_features @ theta
        test_pred = np.argmax(test_pred, axis=1)

        test_labels = test_set["clase"].values
        num_correct = np.sum(test_pred == test_labels)
        return num_correct / len(test_labels)
