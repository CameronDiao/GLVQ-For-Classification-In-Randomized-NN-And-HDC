import scipy as sc
import numpy as np
from numba import njit, vectorize, float64
import kglvq
import math
#import cProfile
#import io
#import pstats

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
    return w_out

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
    #h_matrix = h_matrix.astype(dtype=np.float32)
    #y_matrix = y_matrix.astype(dtype=np.float32)
    w_out = kglvq.KglvqModel()
    w_out.fit(h_matrix, y_matrix)
    return w_out
