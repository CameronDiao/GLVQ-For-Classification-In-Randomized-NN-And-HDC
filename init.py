import numpy as np
import pandas as pd
import sys
sys.path.append("/opt/anaconda3/lib/python3.7/site-packages")
import sklearn_lvq
import hdc

def preprocess(dataset):
    """
    Preprocesses dataset for further analysis
    :param dataset: a DataFrame object containing feature values for every data sample in the object
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

def density_encoding(dataset, n):
    """
    Applies density-based encoding to feature values of dataset
    Encoding is performed by representing quantized feature values as the density of -1s in N-dimensional vectors
    :param dataset: a DataFrame object of dimension M x K containing data samples with their
    respective feature values
    :param n: an int value representing the number of neurons in the network's hidden layer
    :return: data_matrix: a DataFrame object of dimension (MN) x K containing data samples with their
    respective encoded feature values
    """
    # function for quantizing feature values of training data
    v = lambda x_i, n: int(round(x_i * n))
    v = np.vectorize(v)

    # Instantiate empty list of encoded feature values
    data_list = []
    for sample in dataset.index:
        # Apply quantization function to feature values
        x = dataset.loc[sample]
        x = x.to_frame()
        x = v(x, n)
        x = pd.DataFrame(data=np.transpose(x), index=[sample], columns = dataset.loc[sample].index)
        # Encode quantized values as density of -1s in feature vector
        enc_list = []
        for feature in x.columns:
            de = np.zeros(n, dtype = int)
            de[0:x.loc[sample, feature]] = -1
            de[x.loc[sample, feature]:] = 1
            de = pd.DataFrame(data=np.transpose(de))
            enc_list.append(de)
        enc_matrix = pd.concat(enc_list, axis = 1)
        enc_matrix.columns = x.columns
        # Multi-dimensional indexing to map encoded feature vectors to respective samples
        multi_index = [tuple([sample, i]) for i in enc_matrix.index]
        multi_index = pd.MultiIndex.from_tuples(multi_index, names = ['sample', 'dimension'])
        enc_matrix.index = multi_index
        data_list.append(enc_matrix)
    # Construct DataFrame data_matrix of feature value encodings for entire dataset
    data_matrix = pd.concat(data_list, axis = 0)
    return data_matrix


def activations(x, w_in, b, g):
    """
    Computes the activations of the RVFL network's hidden layer h
    :param x: a DataFrame object of dimension K x 1 containing the feature values of one data sample
    :param w_in: a random N x K matrix with each element sampled from a uniform distribution [-1, 1]
    In this instance, a numpy 2-D array
    :param b: a random N x 1 vector with each element sampled from a uniform distribution [-0.1, 0.1]
    In this instance, a numpy 2-D array
    :param g: a nonlinear activation function applied to each neuron
    In this case, the sigmoid function
    :return: h: an N x 1 vector of activation values for one neuron in the network's hidden layer
    In this instance, a numpy 2-D array
    """
    h = w_in.dot(x) + b
    func = np.vectorize(g)
    h = func(h)
    return h

def activation_matrix(dataset, w_in, b, g):
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
    # Instantiate empty list of activation values
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

def enc_activation_matrix(dataset, w_in, kappa):
    """
    Constructs the matrix H of hidden layer activation values from the density-based representation layer
    :param dataset: a DataFrame object of dimension (MN) x K containing data samples with their
    respective encoded feature values
    :param w_in: a random N x K matrix with each element randomly chosen from the set {-1, +1}
    In this instance, a numpy 2-D array
    :param kappa: an int value clipping the bounds of computed activation values
    :return: h_matrix: a DataFrame object of dimension M x N containing the hidden layer activation values
    of every data sample
    """
    # Instantiate empty list of activation values
    h_matrix = []
    for sample, enc in dataset.groupby(level = "sample"):
        # Compute hidden layer activation values for each sample in dataset
        enc_array = enc.to_numpy()
        prod_matrix = hdc.binding(enc_array, w_in)
        h = hdc.bundling(prod_matrix, kappa)
        h = h.reshape((1,dataset.columns.size))
        h = pd.DataFrame(data=h, index=[sample], columns=enc.columns)
        h_matrix.append(h)
    # Construct DataFrame H of activation values for entire dataset
    h_matrix = pd.concat(h_matrix, axis=0)
    return h_matrix


def readout_matrix(h_matrix, y_matrix, lmb):
    """
    Construct matrix w_out of readout connections between the hidden and output layers of the RVFL network
    :param h_matrix: a DataFrame object of dimension M x N containing the hidden layer activation values
    of every data sample
    :param y_matrix: a DataFrame object of dimension M x L containing sample classifications using
    one-hot encodings
    :param lmb: a float value representing the hyperparameter lambda
    :param n: an int value representing the number of neurons in the network's hidden layer
    :return: w_out: a DataFrame object of dimension N x L containing weights of the readout connections
    between the hidden and output layers
    """
    id_matrix = np.diag(np.var(h_matrix.values, axis=0))
    inner = (h_matrix.transpose().dot(h_matrix)).add((id_matrix * lmb))
    inverse_inner = pd.DataFrame(np.linalg.pinv(inner.values), inner.columns, inner.index)
    w_out = inverse_inner.dot(h_matrix.transpose()).dot(y_matrix)
    return w_out

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
    w_out = sklearn_lvq.GlvqModel()
    w_out.fit(h_matrix, y_matrix)
    return w_out

def generate_pred(product_matrix):
    """
    Generate class type predictions based on product_matrix of dot product similarities
    :param product_matrix: a DataFrame object of dimension M x L containing the dot product similarities of
    each data sample to each class type
    :return: pred_series: a Series object containing the class predictions of each data sample
    """
    # Instantiate empty list of class predictions
    pred = []
    # Iterate through every testing sample
    for sample in product_matrix.index:
        # Predict class type of each testing sample based on product_matrix
        sample_pred = product_matrix.loc[sample]
        class_pred = sample_pred.idxmax(axis=1)
        pred.append(class_pred)
    # Convert list of class predictions to a Series object
    pred_series = pd.Series(pred, index=product_matrix.index)
    return pred_series