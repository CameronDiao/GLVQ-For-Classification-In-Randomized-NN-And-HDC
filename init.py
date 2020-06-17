import numpy as np
from numba import jit, vectorize, float64
import sklearn_lvq
import math

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
    dataset = dataset.fillna(1)
    # Add "clase" feature back to dataset
    dataset["clase"] = dataset_class.values
    return dataset

@jit(nopython=True)
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
    for row in range(dataset.shape[0]):
        x = np.zeros(dataset.shape[1])
        for i, elem in np.ndenumerate(dataset[row]):
            v = int(round(elem * n))
            x[i] = v
        enc_matrix = np.ones((x.size, n))
        for vec in range(enc_matrix.shape[0]):
            enc_matrix[vec][0:int(x[vec])] = -1
        enc_ds[row] = enc_matrix.T
    return enc_ds
    # enc_df = dataset.apply(lambda row: de_helper(np.stack(row), n), axis=1)
    # return enc_df

@vectorize([float64(float64)])
def sigmoid(elem):
    return 1 / (1 + math.exp(-elem))

@jit(nopython=True)
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

@jit(nopython=True)
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
    for row in range(dataset.shape[0]):
        x = dataset[row]
        h = np.multiply(x, w_in)
        h = np.sum(h, axis=1)
        for i, elem in np.ndenumerate(h):
            if elem >= kappa:
                h[i] = kappa
            elif elem <= -kappa:
                h[i] = -kappa
        h_matrix[row] = h
    return h_matrix
    # h_matrix = dataset.apply(lambda row: enc_am_helper(np.stack(row), w_in, kappa))
    # return h_matrix

@jit(nopython=True)
def readout_matrix(h_matrix, y_matrix, lmb, n):
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
    id_matrix = np.eye(n)
    w_out = np.linalg.lstsq(h_matrix.T @ h_matrix + id_matrix * lmb, h_matrix.T @ y_matrix)[0]
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
    w_out = sklearn_lvq.RslvqModel()
    w_out.fit(h_matrix, y_matrix)
    return w_out

def generate_pred(product_matrix):
    """
    Generate class type predictions based on product_matrix of dot product similarities
    :param product_matrix: a DataFrame object of dimension M x L containing the dot product similarities of
    each data sample to each class type
    :return: pred_series: a Series object containing the class predictions of each data sample
    """
    return np.argmax(product_matrix, axis=1)
