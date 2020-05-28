import numpy as np
import pandas as pd

def preprocess(dataset):
    """
    Preprocesses dataset for further analysis
    :param dataset: a DataFrame object containing feature values for every data sample
    in the object
    :return: dataset: a preprocessed DataFrame object containing normalized feature values
    for every data sample in the object.
    All missing values in dataset are replaced with 0s
    """
    # Redefine index of train_set
    dataset.set_index("Unnamed: 0", inplace=True)
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

def activations(x, w_in, b, g):
    """
    Computes the activations of the RVFL network's hidden layer h
    :param x: a DataFrame object of dimension K x 1 containing the feature values of one data sample
    :param w_in: a random N x K matrix with each element sampled from a uniform distribution [-1, 1]
    In this instance, a numpy 2-D array.
    :param b: a random N x 1 vector with each element sampled from a uniform distribution [-0.1, 0.1]
    In this instance, a numpy 2-D array.
    :param g: a nonlinear activation function applied to each neuron
    In this case, the sigmoid function is used.
    :return: h: an N x 1 vector of activation values for each neuron in the network's hidden layer
    """
    h = w_in.dot(x) + b
    func = np.vectorize(g)
    h = func(h)
    return h

def activation_matrix(dataset, w_in, b, g):
    """
    Constructs the matrix H of hidden layer activation values for each data sample in dataset
    :param dataset: a DataFrame object of dimension M x N containing data samples with their
    respective feature values
    :param w_in: a random N x K matrix with each element sampled from a uniform distribution [-1, 1]
    In this instance, a numpy 2-D array.
    :param b: a random N x 1 vector with each element sampled from a uniform distribution [-0.1, 0.1]
    In this instance, a numpy 2-D array.
    :param g: a nonlinear activation function applied to each neuron
    In this case, the sigmoid function is used.
    :return: h_matrix: a DataFrame object of dimension M x N containing the hidden layer activation values
    for each data sample in dataset
    """
    # Create empty list of activation values
    h_matrix = []
    for sample in dataset.index:
        # Compute hidden layer activation values for each sample in dataset
        x = dataset.loc[sample]
        x = x.to_frame()
        h = activations(x, w_in, b, g)
        h = pd.DataFrame(data=np.transpose(h), index=[sample])
        h_matrix.append(h)
    # Construct DataFrame H of activation values for entire dataset
    h_matrix = pd.concat(h_matrix, axis=0)
    return h_matrix

def readout_matrix(h_matrix, y_matrix, lmb, n):
    """
    Construct matrix w_out of readout connections between the hidden and output layers of the RVFL network
    :param h_matrix: a DataFrame object of dimension M x N containing the hidden layer activation values
    for each data sample in dataset
    :param y_matrix: a DataFrame object of dimension M x L containing sample classifications using
    one-hot encodings
    :param lmb: a float value representing the hyperparameter lambda
    :param n: an int value representing the number of neurons in the network's hidden layer
    :return: w_out: a DataFrame object of dimension N x L containing weights of the readout connections
    between the hidden and output layers
    """
    id_matrix = np.identity(n)
    inner = (h_matrix.transpose().dot(h_matrix)).add((id_matrix * lmb))
    inverse_inner = pd.DataFrame(np.linalg.pinv(inner.values), inner.columns, inner.index)
    w_out = inverse_inner.dot(h_matrix.transpose()).dot(y_matrix)
    return w_out