from models.ConvRVFL.base import ConvRVFL
from classifiers.lvq1 import lvq1

class ConvRVFLUsingLVQ1(ConvRVFL):
    def __init__(self, train_set, classifier, n, ppc, beta=None, sigma=None):
        super().__init__(train_set, n)
        self.classifier = classifier
        self.ppc = ppc
        self.beta = beta
        self.sigma = sigma

    def readout(self, inputs, labels):
        return lvq1(inputs, labels, self.classifier, ppc=self.ppc, beta=self.beta, sigma=self.sigma)

    def train(self):
        train_features = self.preprocess(self.train_set)
        act_matrix = self.activations(train_features)
        train_labels = self.train_set["clase"].values
        w = self.readout(act_matrix, train_labels)
        return w

    def score(self, test_set):
        w = self.train()
        test_features = self.preprocess(test_set)
        act_matrix = self.activations(test_features)
        test_labels = test_set["clase"].values
        return w.score(act_matrix, test_labels)
