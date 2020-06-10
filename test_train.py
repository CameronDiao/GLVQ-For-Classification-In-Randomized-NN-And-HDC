import numpy as np
import pandas as pd
import math
import init

def test_train_model(train_set, test_set, lmb, n):
    """
    Constructs an RVFL network for predicting the class types of testing data
    :param train_set: a pandas DataFrame containing training samples with their respective feature values
    and class types
    :param test_set: a pandas DataFrame containing testing samples with their respective feature values
    and class types
    :param lmb: a float value representing the hyperparameter lambda
    :param n: an int value representing the number of neurons in the network's hidden layer
    :return: the class type prediction accuracy of the constructed network
    """
    # Preprocess train_set
    #train_set = init.preprocess(train_set)

    # Preprocess test_set
    #test_set = init.preprocess(test_set)

    # Set parameters
    k = len(train_set.columns) - 1
    w_in = np.random.uniform(-1, 1, size=(n, k))
    b = np.random.uniform(-0.1, 0.1, size=(n, 1))
    g = lambda num: 1 / (1 + math.exp(-num))

    # Compute the activation matrix of the hidden layer
    train_features = train_set.drop(["clase"], axis=1)
    h_matrix = init.activation_matrix(train_features, w_in, b, g)
    # Store ground truth classifications of training examples using one-hot encodings
    y_matrix = pd.get_dummies(train_set["clase"])
    # Compute readout matrix from h_matrix, y_matrix
    w_out = init.readout_matrix(h_matrix, y_matrix, lmb)

    # Compute the activation matrix of the hidden layer
    test_features = test_set.drop(["clase"], axis = 1)
    h_matrix = init.activation_matrix(test_features, w_in, b, g)
    # Compute the dot product between h_matrix and w_out
    test_pred = h_matrix.dot(w_out)

    # Generate test_set predictions
    pred_series = init.generate_pred(test_pred)

    # test accuracy
    correct = 0
    for sample in test_set.index:
        if pred_series[sample - 1] == test_set.loc[sample]["clase"]:
            correct += 1
    return correct/len(test_set.index)


def lvq_model(train_set, test_set, n):
    """
    Constructs a neural network employing LVQ classification for predicting the class types of testing samples
    :param train_set: a pandas DataFrame containing training samples with their respective feature values
    and class types
    :param test_set: a pandas DataFrame containing testing samples with their respective feature values
    and class types
    :param n: an int value representing the number of neurons in the network's hidden layer
    :return: the class type prediction accuracy of the constructed network
    """
    # Set parameters
    k = len(train_set.columns) - 1
    w_in = np.random.uniform(-1, 1, size=(n, k))
    b = np.random.uniform(-0.1, 0.1, size=(n, 1))
    g = lambda num: 1 / (1 + math.exp(-num))

    # Compute the activation matrix of the hidden layer
    train_features = train_set.drop(["clase"], axis=1)
    h_matrix = init.activation_matrix(train_features, w_in, b, g)
    # Store ground truth classifications of training examples
    y_matrix = train_set["clase"]
    # Fit an LVQ classifier model to h_matrix, y_matrix
    w_out = init.readout_matrix_lvq(h_matrix, y_matrix)

    # Compute the activation matrix of the hidden layer
    test_features = test_set.drop(["clase"], axis = 1)
    h_matrix = init.activation_matrix(test_features, w_in, b, g)
    # Predict class types using the LVQ classifier model w_out
    #test_pred = w_out.predict(h_matrix)

    # Score prediction accuracy of LVQ classifier model w_out
    return w_out.score(h_matrix, test_set["clase"])

def direct_lvq_model(train_set, test_set):
    """
        Constructs an LVQ classifier model for predicting the class types of testing samples directly from
        feature values
        :param train_set: a pandas DataFrame containing training samples with their respective feature values
        and class types
        :param test_set: a pandas DataFrame containing testing samples with their respective feature values
        and class types
        :return: the class type prediction accuracy of the constructed model
    """
    # Filter out class labels of training samples
    train_features = train_set.drop(["clase"], axis=1)
    # Fit an LVQ classifier model to the feature values of the training data
    w_out = init.readout_matrix_lvq(train_features, train_set["clase"])

    # Compute the activation matrix of the hidden layer
    test_features = test_set.drop(["clase"], axis=1)
    # Predict class types using the LVQ classifier model w_out
    # test_pred = w_out.predict(h_matrix)

    # Score prediction accuracy of LVQ classifier model w_out
    return w_out.score(test_features, test_set["clase"])

def encoding_model(train_set, test_set, lmb, n, kappa):
    """
    Constructs an RVFL network for predicting the class types of testing data
    Operates on density-based encodings of feature values
    :param train_set: a pandas DataFrame containing training samples with their respective feature values
    and class types
    :param test_set: a pandas DataFrame containing testing samples with their respective feature values
    and class types
    :param lmb: a float value representing the hyperparameter lambda
    :param n: an int value representing the number of neurons in the network's hidden layer
    :param kappa: an int value representing the threshold parameter
    :return: the class type prediction accuracy of the constructed network
    """
    # Set parameters
    k = len(train_set.columns) - 1
    w_in = np.random.choice([-1, 1], size=(n, k))

    # Compute density-based representation layer
    train_features = train_set.drop(["clase"], axis=1)
    train_encode_set = init.density_encoding(train_features, n)
    # Compute the activation matrix of the hidden layer
    h_matrix = init.enc_activation_matrix(train_encode_set, w_in, kappa)
    # Store ground truth classifications of training examples using one-hot encodings
    y_matrix = pd.get_dummies(train_set["clase"])
    # Compute readout matrix from h_matrix, y_matrix
    w_out = init.readout_matrix(h_matrix, y_matrix, lmb)

    # Apply density-based encoding to feature values of testing data
    test_features = test_set.drop(["clase"], axis = 1)
    test_encode_set = init.density_encoding(test_features, n)
    # Compute the activation matrix of the hidden layer
    h_matrix = init.enc_activation_matrix(test_encode_set, w_in, kappa)
    # Compute the dot product between h_matrix and w_out
    test_pred = h_matrix.dot(w_out)
    # Generate test_set predictions
    pred_series = init.generate_pred(test_pred)

    # test accuracy
    correct = 0
    for sample in test_set.index:
        if pred_series[sample] == test_set.loc[sample]["clase"]:
            correct += 1

    return correct / len(test_set.index)


def model_accuracy(test_train, lmb, n, kappa):
    """
    Constructs prediction models for every study in test_train
    :param test_train: a dictionary mapping the label of each study parent_name to its
    training and testing datasets
    :param lmb: a float value representing the hyperparameter lambda
    :param n: an int value representing the number of neurons in the network's hidden layer
    :param kappa: an int value representing the threshold parameter
    :return: acc_list: a list of model accuracies for every study in test_train
    """
    # Instantiate empty list of model accuracies
    acc_list = []
    # Determine constructed model accuracies for datasets across all studies
    for name in test_train:
        train_set = test_train[name]["Train"]
        test_set = test_train[name]["Test"]

        acc = encoding_model(train_set, test_set, lmb, n, kappa)
        acc_list.append(acc)
    # Return list of model accuracies
    return acc_list
