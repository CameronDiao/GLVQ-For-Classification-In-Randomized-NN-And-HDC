import numpy as np
import pandas as pd
import init
import torch

def lms_class(train_set, test_set, lmb):
    train_features = train_set.drop(["clase"], axis=1)
    train_features = train_features.values
    train_class = pd.get_dummies(train_set["clase"]).values
    theta = init.readout_matrix(train_features, train_class, lmb)

    test_features = test_set.drop(["clase"], axis=1)
    test_features = test_features.values
    test_pred = test_features @ theta
    pred_series = np.argmax(test_pred, axis=1)

    test_class = test_set["clase"].values
    correct = np.sum(pred_series == test_class)
    return correct / len(test_class)

def lvq_class(train_set, test_set, ppc, beta):
    train_features = train_set.drop(["clase"], axis=1)
    train_features = train_features.values
    w_out = init.readout_matrix_lvq(train_features, train_set["clase"].values, ppc, beta)

    test_features = test_set.drop(["clase"], axis=1)
    test_features = test_features.values

    return w_out.score(test_features, test_set["clase"].values)


def conv_lms(train_set, test_set, lmb, n):
    """
    Constructs a conventional RVFL network for predicting the class types of testing data
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
    b = np.random.uniform(-0.1, 0.1, size = n)

    # Compute the activation matrix of the hidden layer
    train_features = train_set.drop(["clase"], axis=1)
    train_features = train_features.values
    h_matrix = init.activation_matrix(train_features, w_in, b)
    # Store ground truth classifications of training examples using one-hot encodings
    y_matrix = pd.get_dummies(train_set["clase"])
    y_matrix = y_matrix.values.astype(dtype="float64")
    # Compute readout matrix from h_matrix, y_matrix
    w_out = init.readout_matrix(h_matrix, y_matrix, lmb)

    # Compute the activation matrix of the hidden layer
    test_features = test_set.drop(["clase"], axis=1)
    test_features = test_features.values
    h_matrix = init.activation_matrix(test_features, w_in, b)
    # Compute the dot product between h_matrix and w_out
    test_pred = h_matrix @ w_out
    # Generate test_set predictions
    pred_series = np.argmax(test_pred, axis=1)

    # test accuracy
    test_class = test_set["clase"].values
    correct = np.sum(pred_series == test_class)
    return correct / len(test_class)

def conv_lvq(train_set, test_set, n, ppc, beta):
    # Set parameters
    k = len(train_set.columns) - 1
    w_in = np.random.uniform(-1, 1, size=(n, k))
    b = np.random.uniform(-0.1, 0.1, size=n)

    train_features = train_set.drop(["clase"], axis=1)
    train_features = train_features.values
    h_matrix = init.activation_matrix(train_features, w_in, b)
    w_out = init.readout_matrix_lvq(h_matrix, train_set["clase"].values, ppc, beta)

    test_features = test_set.drop(["clase"], axis=1)
    test_features = test_features.values
    h_matrix = init.activation_matrix(test_features, w_in, b)
    return w_out.score(h_matrix, test_set["clase"].values)

def lvq_model(train_set, test_set, n, kappa, ppc, beta):
    """
    Constructs an RVFL network employing LVQ classification for predicting the class types of testing samples
    Uses the sklearn_lvq package for LVQ classification.
    :param train_set: a pandas DataFrame containing training samples with their respective feature values
    and class types
    :param test_set: a pandas DataFrame containing testing samples with their respective feature values
    and class types
    :param n: an int value representing the number of neurons in the network's hidden layer
    :param kappa: an int value representing the threshold parameter
    :param ppc: an int value representing the number of prototype vectors per class
    :param beta: an int value representing the hyperparameter beta
    :return: the class type prediction accuracy of the constructed network
    """
    # Set parameters
    k = len(train_set.columns) - 1
    w_in = np.random.choice([-1, 1], size=(n, k)).astype(dtype="float64")

    # Compute density-based representation layer
    train_features = train_set.drop(["clase"], axis=1)
    train_features = train_features.values
    train_encode_set = init.density_encoding(train_features, n)
    # Compute the activation matrix of the hidden layer
    h_matrix = init.enc_activation_matrix(train_encode_set, w_in, kappa)
    # Store ground truth classifications of training examples
    y_matrix = train_set["clase"].values
    # Compute readout matrix from h_matrix, y_matrix
    w_out = init.readout_matrix_lvq(h_matrix, y_matrix, ppc, beta)

    # Apply density-based encoding to feature values of testing data
    test_features = test_set.drop(["clase"], axis=1)
    test_features = test_features.values
    test_encode_set = init.density_encoding(test_features, n)
    # Compute the activation matrix of the hidden layer
    h_matrix = init.enc_activation_matrix(test_encode_set, w_in, kappa)
    # Predict class types using the LVQ classifier model w_out
    #test_pred = w_out.predict(h_matrix)

    # Score prediction accuracy of LVQ classifier model w_out
    return w_out.score(h_matrix, test_set["clase"].values)


def lvq_model2(train_set, test_set, n, kappa, ppc, beta):
    """
    Constructs an RVFL employing LVQ classification for predicting the class types of testing samples
    Uses prototorch package for LVQ classification.
    :param train_set: a pandas DataFrame containing training samples with their respective feature values
    and class types
    :param test_set: a pandas DataFrame containing testing samples with their respective feature values
    and class types
    :param n: an int value representing the number of neurons in the network's hidden layer
    :param kappa: an int value representing the threshold parameter
    :param ppc: an int value representing the number of prototype vectors per class
    :param beta: an int value representing the hyperparameter beta
    :return: the class type prediction accuracy of the constructed network
    """
    # Set parameters
    k = len(train_set.columns) - 1
    w_in = np.random.choice([-1, 1], size=(n, k)).astype(dtype="float64")

    # Compute density-based representation layer
    train_features = train_set.drop(["clase"], axis=1)
    train_features = train_features.values
    train_encode_set = init.density_encoding(train_features, n)
    # Compute the activation matrix of the hidden layer
    h_matrix = init.enc_activation_matrix(train_encode_set, w_in, kappa)
    # Store ground truth classifications of training examples
    y_matrix = train_set["clase"].values
    # Fit an LVQ classifier model to the activation values of h_matrix
    w_out = init.readout_matrix_lvq2(h_matrix, y_matrix, ppc, beta)

    # Apply density-based encoding to feature values of testing data
    test_features = test_set.drop(["clase"], axis=1)
    test_features = test_features.values
    test_encode_set = init.density_encoding(test_features, n)
    # Compute the activation matrix of the hidden layer
    h_matrix = init.enc_activation_matrix(test_encode_set, w_in, kappa)
    # Collect ground truth classifications
    test_class = test_set["clase"].values
    # Run LVQ Classifier on h_matrix activation values of test set
    with torch.no_grad():
        w_out.eval()
        distances, plabels = w_out(torch.from_numpy(h_matrix))
        _, prediction = torch.min(distances.data, 1)
        prediction = torch.floor_divide(prediction, ppc)
        test_acc = torch.sum(prediction == torch.from_numpy(test_class))
    return test_acc.item() / len(test_class)

def direct_lvq_model(train_set, test_set, ppc, beta):
    """
    Constructs an LVQ classifier model for predicting the class types of testing samples directly from
    feature values
    :param train_set: a pandas DataFrame containing training samples with their respective feature values
    and class types
    :param test_set: a pandas DataFrame containing testing samples with their respective feature values
    and class types
    :param ppc: an int value representing the number of prototype vectors per class
    :param beta: an int value representing the hyperparameter beta
    :return: the class type prediction accuracy of the constructed model
    """
    # Filter out class labels of training samples
    train_features = train_set.drop(["clase"], axis=1)
    train_features = train_features.values
    # Fit an LVQ classifier model to the feature values of the training data
    w_out = init.readout_matrix_lvq(train_features, train_set["clase"].values, ppc, beta)

    # Compute the activation matrix of the hidden layer
    test_features = test_set.drop(["clase"], axis=1)
    test_features = test_features.values
    # Predict class types using the LVQ classifier model w_out
    # test_pred = w_out.predict(h_matrix)

    # Score prediction accuracy of LVQ classifier model w_out
    return w_out.score(test_features, test_set["clase"].values)

def direct_lvq_model2(train_set, test_set, ppc, beta, sigma):
    # Filter out class labels of training samples
    train_features = train_set.drop(["clase"], axis=1)
    train_features = train_features.values
    # Fit an LVQ classifier model to the feature values of the training data
    w_out = init.readout_matrix_kglvq(train_features, train_set["clase"].values, ppc, beta, sigma)

    # Compute the activation matrix of the hidden layer
    test_features = test_set.drop(["clase"], axis=1)
    test_features = test_features.values
    # Predict class types using the LVQ classifier model w_out
    # test_pred = w_out.predict(h_matrix)
    test_class = test_set["clase"].values
    # Run LVQ Classifier on h_matrix activation values of test set
    with torch.no_grad():
        w_out.eval()
        distances, plabels = w_out(torch.from_numpy(test_features))
        _, prediction = torch.min(distances.data, 1)
        prediction = torch.floor_divide(prediction, ppc)
        test_acc = torch.sum(prediction == torch.from_numpy(test_class))
    return test_acc.item() / len(test_class)

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
    w_in = np.random.choice([-1, 1], size=(n, k)).astype(dtype="float64")

    # Compute density-based representation layer
    train_features = train_set.drop(["clase"], axis=1)
    train_features = train_features.values
    train_encode_set = init.density_encoding(train_features, n)
    # Compute the activation matrix of the hidden layer
    h_matrix = init.enc_activation_matrix(train_encode_set, w_in, kappa)
    # Store ground truth classifications of training examples using one-hot encodings
    y_matrix = pd.get_dummies(train_set["clase"])
    # Compute readout matrix from h_matrix, y_matrix
    w_out = init.readout_matrix(h_matrix, y_matrix, lmb)

    # Apply density-based encoding to feature values of testing data
    test_features = test_set.drop(["clase"], axis = 1)
    test_features = test_features.values
    test_encode_set = init.density_encoding(test_features, n)
    # Compute the activation matrix of the hidden layer
    h_matrix = init.enc_activation_matrix(test_encode_set, w_in, kappa)
    # Compute the dot product between h_matrix and w_out
    test_pred = h_matrix.astype(dtype=np.float32) @ w_out
    # Generate test_set predictions
    pred_series = np.argmax(test_pred, axis=1)

    # test accuracy
    test_class = test_set["clase"].values
    correct = np.sum(pred_series == test_class)
    return correct / len(test_class)