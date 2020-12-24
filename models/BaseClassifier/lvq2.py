import torch

from models.BaseClassifier.base import BaseClassifier
from classifiers.lvq2 import lvq2

class LVQClassifier2(BaseClassifier):
    def __init__(self, train_set, classifier, optimizer, ppc, beta, sigma=None):
        super().__init__(train_set)
        self.classifier = classifier
        self.optimizer = optimizer
        self.ppc = ppc
        self.beta = beta
        self.sigma = sigma

    def model(self, inputs, labels):
        return lvq2(inputs, labels, self.classifier, self.optimizer, ppc=self.ppc, beta=self.beta, sigma=self.sigma)

    def train(self):
        train_features = self.preprocess(self.train_set)
        train_labels = self.train_set["clase"].values
        w = self.model(train_features, train_labels)
        return w

    def score(self, test_set):
        w = self.train()
        test_features = self.preprocess(test_set)
        test_labels = test_set["clase"].values
        with torch.no_grad():
            w.eval()
            distances, plabels = w(torch.from_numpy(test_features))
            _, prediction = torch.min(distances.data, 1)
            prediction = torch.floor_divide(prediction, self.ppc)
            test_acc = torch.sum(prediction == torch.from_numpy(test_labels))
        return test_acc.item() / len(test_labels)
