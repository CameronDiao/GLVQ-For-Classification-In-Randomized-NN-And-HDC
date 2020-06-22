import numpy as np
import scipy as sc
from numba import njit, vectorize, float64, prange
import sklearn_lvq
import math
import glvq
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
    Encoding is performed by representing quantized feature values as the density of -1s in N-dimensional vectors
    :param dataset: a data matrix of dimension M x K containing data samples with their
    respective feature values
    In this instance, a DataFrame object of shape M x K
    :param n: an int value representing the number of neurons in the network's hidden layer
    :return: data_matrix: a matrix of dimension (MN) x K containing data samples with their
    respective encoded feature values
    In this instance, a Series object of size M containing numpy 2-D arrays of shape N x K
    """
    enc_ds = np.zeros((dataset.shape[0], n, dataset.shape[1]))
    for row in prange(dataset.shape[0]):
        x = quantize(dataset[row], np.float64(n))
        enc_matrix = np.ones((x.size, n))
        for vec in prange(enc_matrix.shape[0]):
            enc_matrix[vec][0:int(x[vec])] = -1
        enc_ds[row] = enc_matrix.T
    return enc_ds
    # enc_df = dataset.apply(lambda row: de_helper(np.stack(row), n), axis=1)
    # return enc_df

@vectorize([float64(float64)])
def sigmoid(elem):
    return 1 / (1 + math.exp(-elem))

@njit
def activation_matrix(dataset, w_in, b):
    """
    Constructs the matrix H of hidden layer activation values of every data sample in dataset
    :param dataset: a DataFrame object of dimension M x K containing data samples with their
    respective feature values
    :param w_in: a random N x K matrix with each element sampled from a uniform distribution [-1, 1]
    In this instance, a numpy 2-D array
    :param b: a random N x 1 vector with each element sampled from a uniform distribution [-0.1, 0.1]
    In this instance, a numpy 2-D array
    :param g: a nonlinear activation function applied to each neuron
    In this case, the sigmoid function is used
    :return: h_matrix: a DataFrame object of dimension M x N containing the hidden layer activation values
    of every data sample
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
    In this instance, a Series object of size M containing numpy 2-D arrays of shape N x K
    :param w_in: a random N x K matrix with each element randomly chosen from the set {-1, +1}
    In this instance, a numpy 2-D array
    :param kappa: an int value clipping the bounds of computed activation values
    :return: h_matrix: a data matrix of dimension M x N containing the hidden layer activation values
    of every data sample
    In this instance, a Series object of size M containing numpy 1-D arrays of size N
    """
    h_matrix = np.zeros((dataset.shape[0], dataset.shape[1]))
    for row in prange(dataset.shape[0]):
        x = dataset[row]
        h = np.sum(np.multiply(x, w_in), axis=1)
        for i, elem in np.ndenumerate(h):
            if elem >= kappa:
                h[i] = kappa
            elif elem <= -kappa:
                h[i] = -kappa
        h_matrix[row] = h
    return h_matrix
    # h_matrix = dataset.apply(lambda row: enc_am_helper(np.stack(row), w_in, kappa))
    # return h_matrix

def readout_matrix(h_matrix, y_matrix, lmb):
    """
    Construct matrix w_out of readout connections between the hidden and output layers of the RVFL network
    :param h_matrix: a data matrix of dimension M x N containing the hidden layer activation values
    of every data sample
    In this instance, a numpy 2-D array of shape M x N
    :param y_matrix: a data matrix of dimension M x L containing sample classifications using
    one-hot encodings
    In this instance, a DataFrame object of shape M x L
    :param lmb: a float value representing the hyperparameter lambda
    :param n: an int value representing the number of neurons in the network's hidden layer
    :return: w_out: a data matrix of dimension N x L containing weights of the readout connections
    between the hidden and output layers
    In this instance, a numpy 2-D array of shape N x L
    """
    h_matrix = h_matrix.astype(dtype=np.float32)
    y_matrix = y_matrix.astype(dtype=np.float32)
    id_matrix = np.diag(np.var(h_matrix, axis=0))
    #w_out = np.linalg.lstsq(h_matrix.T @ h_matrix + id_matrix * lmb, h_matrix.T @ y_matrix)[0]
    w_out = sc.linalg.pinv(h_matrix.T @ h_matrix + id_matrix * lmb) @ h_matrix.T @ y_matrix
    return w_out
    #inner = np.add(x.T @ x, id_matrix * lmb)
    #w_out = np.linalg.pinv(inner) @ x.T @ y_matrix
    #w_out = np.linalg.lstsq(x.T @ x + id_matrix * lmb, x.T @ y_matrix, rcond=None)[0]
    #return w_out

def readout_matrix_lvq(h_matrix, y_matrix):
    """
    Fit an LVQ classifier model from h_matrix and y_matrix
    :param h_matrix: a DataFrame object of dimension M x N containing the hidden layer activation values
    of every data sample
    Alternatively, a DataFrame object of dimension M x K containing the normalized feature values
    of every data sample
    :param y_matrix: a DataFrame object of dimension M x 1 containing sample classifications
    :return: w_out: an LVQ Model object
    """
    #h_matrix = h_matrix.astype(dtype=np.float32)
    #y_matrix = y_matrix.astype(dtype=np.float32)
    w_out = glvq.GlvqModel(prototypes_per_class=3, beta=13)
    #pr = cProfile.Profile()
    #pr.enable()
    w_out.fit(h_matrix, y_matrix)
    #pr.disable()
    #s = io.StringIO()
    #sortby = pstats.SortKey.CUMULATIVE  # 'cumulative'
    #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.print_stats()
    #print(s.getvalue())
    return w_out