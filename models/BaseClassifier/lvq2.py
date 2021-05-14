import numpy as np
import torch
from torch.utils.data import TensorDataset

from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel
from classifiers.pytorch import kernel_distance

from models.BaseClassifier.base import BaseClassifier
from classifiers import lvq2

class LVQClassifier2(BaseClassifier):
    def __init__(self, train_set, classifier, optimizer, epochs, ppc, beta, sigma=None):
        super().__init__(train_set)
        self.classifier = classifier
        self.optimizer = optimizer
        self.ppc = ppc
        self.beta = beta
        self.sigma = round(sigma, 1)
        self.epochs = epochs

    def model(self, inputs, labels):
        return lvq2(inputs, labels, self.classifier, self.optimizer, epochs=self.epochs,
                    ppc=self.ppc, beta=self.beta, sigma=self.sigma)

    def train(self):
        self.train_features = self.preprocess(self.train_set)
        train_labels = self.train_set["clase"].values
        w = self.model(self.train_features, train_labels)
        return w

    def score(self, test_set, classifier=None, wrapper=None):
        w = self.train()
        test_features = self.preprocess(test_set)
        test_labels = test_set["clase"].values
        if wrapper:
            if classifier == 'glvq':
                dist = cdist(test_features, w, 'sqeuclidean')
            elif classifier == 'kglvq':
                dist = kernel_distance(torch.from_numpy(rbf_kernel(test_features, gamma=self.sigma)),
                                       torch.from_numpy(rbf_kernel(test_features, self.train_features, gamma=self.sigma)),
                                       torch.from_numpy(rbf_kernel(self.train_features, gamma=self.sigma)),
                                       torch.from_numpy(test_features), torch.from_numpy(w)).numpy()
            else:
                raise ValueError("Invalid classifier")
            test_acc = np.sum(test_labels == np.floor_divide(dist.argmin(1), self.ppc))
            return test_acc / len(test_labels)
        test_features = torch.from_numpy(test_features)
        test_labels = torch.from_numpy(test_labels)

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        w.to(device)
        testloader = torch.utils.data.DataLoader(TensorDataset(test_features, test_labels), batch_size=16,
                                                 num_workers=0)
        test_acc = torch.tensor(0)
        with torch.no_grad():
            w.eval()
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                distances, plabels = w(inputs)
                _, prediction = torch.min(distances, 1)
                prediction = torch.floor_divide(prediction, self.ppc)
                test_acc = test_acc + torch.sum(prediction == targets)
        return test_acc.item() / len(test_labels)
