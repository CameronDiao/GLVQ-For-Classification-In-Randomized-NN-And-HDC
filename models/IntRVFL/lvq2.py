import numpy as np
import torch
from torch.utils.data import TensorDataset

from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel
from classifiers.pytorch import kernel_distance

from .base import IntRVFL
from classifiers import lvq2

class IntRVFLUsingLVQ2(IntRVFL):
    def __init__(self, train_set, classifier, optimizer, epochs, n, kappa, ppc, beta, sigma=None):
        super().__init__(train_set, n, kappa)
        self.classifier = classifier
        self.optimizer = optimizer
        self.ppc = ppc
        self.beta = beta
        self.sigma = sigma
        self.epochs = epochs

    def readout(self, inputs, labels):
        return lvq2(inputs, labels, self.classifier, self.optimizer, epochs=self.epochs,
                    ppc=self.ppc, beta=self.beta, sigma=self.sigma)

    def train(self):
        train_features = self.preprocess(self.train_set)
        enc_matrix = self.encodings(train_features)
        self.act_matrix = self.activations(enc_matrix)
        train_labels = self.train_set["clase"].values
        w = self.readout(self.act_matrix, train_labels)
        return w

    def score(self, test_set, classifier=None, wrapper=None):
        w = self.train()
        test_features = self.preprocess(test_set)
        enc_matrix = self.encodings(test_features)
        act_matrix = self.activations(enc_matrix)
        test_labels = test_set["clase"].values
        if wrapper:
            if classifier == 'glvq':
                dist = cdist(act_matrix, w, 'sqeuclidean')
            elif classifier == 'kglvq':
                dist = kernel_distance(torch.from_numpy(rbf_kernel(act_matrix, gamma=self.sigma)),
                                       torch.from_numpy(rbf_kernel(act_matrix, self.act_matrix, gamma=self.sigma)),
                                       torch.from_numpy(rbf_kernel(self.act_matrix, gamma=self.sigma)),
                                       torch.from_numpy(act_matrix), torch.from_numpy(w)).numpy()
            else:
                raise ValueError("Invalid classifier")
            test_acc = np.sum(test_labels == np.floor_divide(dist.argmin(1), self.ppc))
            return test_acc / len(test_labels)

        act_matrix = torch.from_numpy(act_matrix)
        test_labels = torch.from_numpy(test_labels)

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        w.to(device)
        testloader = torch.utils.data.DataLoader(TensorDataset(act_matrix, test_labels), batch_size=32, num_workers=0)
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