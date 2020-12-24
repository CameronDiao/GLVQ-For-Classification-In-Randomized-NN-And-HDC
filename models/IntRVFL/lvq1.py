from models.IntRVFL.base import IntRVFL
from classifiers import lvq1

class IntRVFLUsingLVQ1(IntRVFL):
    def __init__(self, train_set, classifier, n, kappa, ppc, beta=None, sigma=None):
        super().__init__(train_set, n, kappa)
        self.classifier = classifier
        self.ppc = ppc
        self.beta = beta
        self.sigma = sigma

    def readout(self, inputs, labels):
        return lvq1(inputs, labels, self.classifier, ppc=self.ppc, beta=self.beta, sigma=self.sigma)

    def train(self):
        train_features = self.preprocess(self.train_set)
        enc_matrix = self.encodings(train_features)
        act_matrix = self.activations(enc_matrix)
        train_labels = self.train_set["clase"].values
        w = self.readout(act_matrix, train_labels)
        return w

    def score(self, test_set):
        w = self.train()
        test_features = self.preprocess(test_set)
        enc_matrix = self.encodings(test_features)
        act_matrix = self.activations(enc_matrix)
        test_labels = test_set["clase"].values
        return w.score(act_matrix, test_labels)
