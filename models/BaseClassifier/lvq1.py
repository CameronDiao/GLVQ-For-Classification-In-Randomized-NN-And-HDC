from .base import BaseClassifier
from classifiers import lvq1

class LVQClassifier1(BaseClassifier):
    def __init__(self, train_set, classifier, epochs, ppc, beta=None, sigma=None):
        super().__init__(train_set)
        self.classifier = classifier
        self.ppc = ppc
        self.beta = beta
        self.sigma = sigma
        self.epochs = epochs

    def model(self, inputs, labels):
        return lvq1(inputs, labels, self.classifier, epochs=self.epochs, ppc=self.ppc, beta=self.beta, sigma=self.sigma)

    def train(self):
        train_features = self.preprocess(self.train_set)
        train_labels = self.train_set["clase"].values
        w = self.model(train_features, train_labels)
        return w

    def score(self, test_set):
        w = self.train()
        test_features = self.preprocess(test_set)
        test_labels = test_set["clase"].values
        return w.score(test_features, test_labels)
