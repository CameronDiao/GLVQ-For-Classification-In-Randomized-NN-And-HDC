import math
import scipy as sc
import numpy as np
from numba import njit, vectorize, float64
import sklearn_lvq
#import cProfile
#import io
#import pstats

import torch
from torch.utils.data import TensorDataset
from sklearn.metrics.pairwise import rbf_kernel
from distances import euclidean_distance, kernel_distance, nystroem_kernel_distance
from loss import GLVQLoss, KGLVQLoss
from prototypes import Prototypes1D

def preprocess(dataset):
    """
    Preprocesses dataset for further analysis
    :param dataset: a DataFrame object containing feature values for every data sample in the object
    :return: dataset: a preprocessed DataFrame object containing normalized feature values
    for every data sample in the object.
    All missing values in dataset are replaced with 0s
    """
    # Redefine index of train_set
    dataset = dataset.drop(["Unnamed: 0"], axis=1)
    # Exclude "clase" feature from pre-processing
    dataset_class = dataset["clase"]
    dataset = dataset.drop(["clase"], axis=1)
    # Normalize features in dataset to the range [0, 1]
    dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    # Fill missing values with 0s
    dataset = dataset.fillna(0)
    # Add "clase" feature back to dataset
    dataset["clase"] = dataset_class.values
    return dataset

@vectorize([float64(float64, float64)])
def quantize(elem, n):
    return int(round(elem * n))

@njit
def density_encoding(dataset, n):
    """
    Applies density-based encoding to feature values of dataset
    Encoding is performed by representing quantized feature values as the density of N-dimensional bipolar vectors
    :param dataset: a data matrix of dimension M x K containing data samples with their
    respective feature values
    In this instance, a numpy 2-D array
    :param n: an int value representing the number of neurons in the network's hidden layer
    :return: data_matrix: a matrix of dimension (MN) x K containing data samples with their
    respective encoded feature values
    In this instance, a numpy 3-D array of shape M x N x K
    """
    enc_ds = np.zeros((dataset.shape[0], n, dataset.shape[1]))
    for row in range(dataset.shape[0]):
        x = quantize(dataset[row], np.float64(n))
        enc_matrix = np.ones((x.size, n))
        for vec in range(enc_matrix.shape[0]):
            enc_matrix[vec][0:int(x[vec])] = -1
        enc_ds[row] = enc_matrix.T
    return enc_ds

@vectorize([float64(float64)])
def sigmoid(elem):
    return 1 / (1 + math.exp(-elem))

@njit
def activation_matrix(dataset, w_in, b):
    """
    Constructs the matrix H of hidden layer activation values of every data sample in dataset
    :param dataset: a data matrix of dimension M x K containing data samples with their
    respective feature values
    In this instance, a numpy 2-D array
    :param w_in: a random N x K matrix with each element sampled from a uniform distribution [-1, 1]
    In this instance, a numpy 2-D array
    :param b: a random N x 1 vector with each element sampled from a uniform distribution [-0.1, 0.1]
    In this instance, a numpy 2-D array
    :return: h_matrix: a data matrix of dimension M x N containing the hidden layer activation values
    of every data sample
    In this instance, a numpy 2-D array
    """
    #h_matrix = np.zeros((dataset.shape[0], n))
    #for row in range(dataset.shape[0]):
    #    x = dataset[row]
    #    h = w_in @ x + b
    #    h = sigmoid(h)
    #    h_matrix[row] = h
    h_matrix = dataset @ w_in.T
    h_matrix = h_matrix + b
    h_matrix = sigmoid(h_matrix)
    return h_matrix

@njit
def enc_activation_matrix(dataset, w_in, kappa):
    """
    Constructs the matrix H of hidden layer activation values from the density-based representation layer
    :param dataset: a data matrix of dimension (MN) x K containing data samples with their
    respective encoded feature values
    In this instance, a numpy 2-D array of shape M x N x K
    :param w_in: a random N x K matrix with each element randomly chosen from the set {-1, +1}
    In this instance, a numpy 2-D array
    :param kappa: an int value clipping the bounds of computed activation values
    :return: h_matrix: a data matrix of dimension M x N containing the hidden layer activation values
    of every data sample
    In this instance, a numpy 2-D array
    """
    h_matrix = np.zeros((dataset.shape[0], dataset.shape[1]))
    for row in range(dataset.shape[0]):
        x = dataset[row]
        h = np.sum(np.multiply(x, w_in), axis=1)
        for i, elem in np.ndenumerate(h):
            if elem >= kappa:
                h[i] = kappa
            elif elem <= -kappa:
                h[i] = -kappa
        h_matrix[row] = h
    return h_matrix

def readout_matrix(h_matrix, y_matrix, lmb):
    """
    Constructs matrix w_out of readout connections between the hidden and output layers of the RVFL network
    :param h_matrix: a data matrix of dimension M x N containing the hidden layer activation values
    of every data sample
    In this instance, a numpy 2-D array
    :param y_matrix: a data matrix of dimension M x L containing sample classifications using
    one-hot encodings
    In this instance, a numpy 2-D array
    :param lmb: a float value representing the hyperparameter lambda
    :return: w_out: a data matrix of dimension N x L containing weights of the readout connections
    between the hidden and output layers
    In this instance, a numpy 2-D array
    """
    h_matrix = h_matrix.astype(dtype=np.float32)
    y_matrix = y_matrix.astype(dtype=np.float32)
    id_matrix = np.diag(np.var(h_matrix, axis=0))
    #w_out = np.linalg.lstsq(h_matrix.T @ h_matrix + id_matrix * lmb, h_matrix.T @ y_matrix)[0]
    w_out = sc.linalg.pinv(h_matrix.T @ h_matrix + id_matrix * lmb) @ h_matrix.T @ y_matrix
    return w_out.values

def readout_matrix_lvq(h_matrix, y_matrix, ppc, beta):
    """
    Fit a KGLVQ classifier model from h_matrix and y_matrix
    :param h_matrix: a data matrix of dimension M x N containing the hidden layer activation values
    of every data sample
    Alternatively, a data matrix of dimension M x K containing the normalized feature values
    of every data sample
    In this instance, a numpy 2-D array
    :param y_matrix: a data matrix of dimension M x 1 containing sample classifications
    In this instance, a numpy 1-D array
    :return: w_out: an LvqModel object
    """
    h_matrix = h_matrix.astype(dtype=np.float32)
    y_matrix = y_matrix.astype(dtype=np.float32)
    w_out = sklearn_lvq.GlvqModel(prototypes_per_class=ppc, beta=beta)
    w_out.fit(h_matrix, y_matrix)
    return w_out

def readout_matrix_lvq2(h_matrix, y_matrix, ppc, beta):
    h_matrix = torch.from_numpy(h_matrix).float()
    y_matrix = torch.from_numpy(y_matrix).float()

    epochs = 200

    model = glvq_module(h_matrix, y_matrix, ppc=ppc)

    optimizer = torch.optim.SGD(model.parameters(), lr = 1.0)
    criterion = GLVQLoss(squashing="sigmoid_beta", beta = beta)

    full_train(h_matrix, y_matrix, model, epochs, optimizer, criterion)
    model.load_state_dict(torch.load('/Users/camerondiao/Documents/HDResearch/DataManip/checkpoint.pt'))
    return model

def readout_matrix_kglvq(h_matrix, y_matrix, ppc, beta, sigma):
    h_matrix = torch.from_numpy(h_matrix).float()
    y_matrix = torch.from_numpy(y_matrix).float()

    epochs = 50

    model = kglvq_module(h_matrix, y_matrix, ppc=ppc, sigma=sigma)
    #model = kglvq_module(h_matrix, y_matrix, ppc=ppc, sigma=sigma)

    optimizer = torch.optim.LBFGS(model.parameters())
    criterion = KGLVQLoss(squashing='sigmoid_beta', beta=beta)

    full_train(h_matrix, y_matrix, model, epochs, optimizer, criterion)
    #batch_train(h_matrix, y_matrix, model, epochs, optimizer, criterion)
    model.load_state_dict(torch.load('/Users/camerondiao/Documents/HDResearch/DataManip/checkpoint.pt'))
    return model

def glvq_module(x_data, y_data, ppc):
    class Model(torch.nn.Module):
        def __init__(self, x_data, y_data, **kwargs):
            super().__init__()
            self.p1 = Prototypes1D(input_dim=x_data.shape[1],
                                   prototypes_per_class=ppc,
                                   nclasses=torch.unique(y_data).size()[0],
                                   prototype_initializer='stratified_mean',
                                   data=[x_data, y_data])
            self.train_data = x_data
        def forward(self, x):
            protos = self.p1.prototypes
            plabels = self.p1.prototype_labels
            dis = euclidean_distance(x, protos)
            return dis, plabels

    return Model(x_data=x_data, y_data=y_data)

def kglvq_module(x_data, y_data, ppc, sigma=None):
    class Model(torch.nn.Module):
        def __init__(self, x_data, y_data, **kwargs):
            super().__init__()
            self.p1 = Prototypes1D(input_dim=x_data.shape[1],
                                   prototypes_per_class=ppc,
                                   nclasses=torch.unique(y_data).size()[0],
                                   prototype_initializer='kernel_mean',
                                   data=[x_data, y_data])
            self.train_data = x_data

        def forward(self, x):
            protos = self.p1.prototypes
            plabels = self.p1.prototype_labels
            dis = kernel_distance(torch.from_numpy(rbf_kernel(x, gamma=sigma)),
                                  torch.from_numpy(rbf_kernel(x, self.train_data, gamma=sigma)),
                                  torch.from_numpy(rbf_kernel(self.train_data, gamma=sigma)), x, protos)
            return dis, plabels

    return Model(x_data=x_data, y_data=y_data)

def akglvq_module(x_data, y_data, ppc, sigma=None):
    class Model(torch.nn.Module):
        def __init__(self, x_data, y_data, **kwargs):
            super().__init__()
            self.p1 = Prototypes1D(input_dim=x_data.shape[1],
                                   prototypes_per_class=ppc,
                                   nclasses=torch.unique(y_data).size()[0],
                                   prototype_initializer='kernel_mean',
                                   data=[x_data, y_data])
            self.train_data = x_data
            self.train_samples = x_data[torch.randperm(x_data.size(0))[:int(x_data.size(0) / 5)]]

        def forward(self, x):
            protos = self.p1.prototypes
            plabels = self.p1.prototype_labels

            q = rbf_kernel(self.train_samples, self.train_samples, gamma=sigma)
            n = rbf_kernel(self.train_data, self.train_samples, gamma=sigma)

            dis = nystroem_kernel_distance(torch.from_numpy(rbf_kernel(x, self.train_data, gamma=sigma)),
                                           torch.from_numpy(q),
                                           torch.from_numpy(n), x, protos)
            return dis, plabels

    return Model(x_data=x_data, y_data=y_data)

def full_train(x_data, y_data, model, epochs, optimizer, criterion):
    best_loss = np.inf
    #past_loss = None
    #delta = 1e-5

    for epoch in range(epochs):
        model.train()
        def closure():
            optimizer.zero_grad()
            distances, plabels = model(x_data)
            loss = criterion([distances, plabels], y_data)
            #print(f'Epoch: {epoch + 1:03d} Loss: {loss.item():02.02f}')
            loss.backward()
            return loss

        optimizer.step(closure)

        model.eval()
        with torch.no_grad():
            distances, plabels = model(x_data)
            loss = criterion([distances, plabels], y_data)

        #if past_loss is None or abs(past_loss - loss) > delta:
        #    past_loss = loss

        if best_loss > loss:
            best_loss = loss
            torch.save(model.state_dict(), 'checkpoint.pt')

def batch_train(x_data, y_data, model, epochs, optimizer, criterion):
    trainset = TensorDataset(x_data, y_data)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, num_workers=0,
                                              shuffle=True)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in trainloader:
            optimizer.zero_grad()
            distances, plabels = model(inputs)
            loss = criterion([distances, plabels], targets)
            print(f'Epoch: {epoch + 1:03d} Loss: {loss.item():02.02f}')
            loss.backward()
            optimizer.step()
