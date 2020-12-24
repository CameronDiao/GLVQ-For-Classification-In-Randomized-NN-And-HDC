import torch

from models.ConvRVFL.base import ConvRVFL
from classifiers import lvq2

class ConvRVFLUsingLVQ2(ConvRVFL):
    def __init__(self, train_set, classifier, optimizer, n, ppc, beta, sigma=None):
        super().__init__(train_set, n)
        self.classifier = classifier
        self.optimizer = optimizer
        self.ppc = ppc
        self.beta = beta
        self.sigma = sigma

    def readout(self, inputs, labels):
        return lvq2(inputs, labels, self.classifier, self.optimizer, ppc=self.ppc, beta=self.beta, sigma=self.sigma)

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
        with torch.no_grad():
            w.eval()
            distances, plabels = w(torch.from_numpy(act_matrix))
            _, prediction = torch.min(distances.data, 1)
            prediction = torch.floor_divide(prediction, self.ppc)
            test_acc = torch.sum(prediction == torch.from_numpy(test_labels))
        return test_acc.item() / len(test_labels)
