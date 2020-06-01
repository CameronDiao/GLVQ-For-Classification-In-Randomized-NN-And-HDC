import numpy as np
import pandas as pd
import math
import init
import pred

def test_train_model(train_set, test_set, lmb, n):
    """
    Constructs a neural network for predicting the class types of testing samples
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
    #y_matrix = pd.get_dummies(train_set["clase"])
    y_matrix = train_set["clase"]
    # Fit a GLVQ classifier model to h_matrix, y_matrix
    w_out = init.readout_matrix_glvq(h_matrix, y_matrix)

    # Compute the activation matrix of the hidden layer
    test_features = test_set.drop(["clase"], axis = 1)
    h_matrix = init.activation_matrix(test_features, w_in, b, g)
    # Compute the dot product between h_matrix and w_out
    #test_pred = h_matrix.dot(w_out)
    # Predict class types using the GLVQ classifier model w_out
    #test_pred = w_out.predict(h_matrix)

    # test accuracy
    #correct = 0
    #for sample in train_set.index:
    #    if test_pred[sample - 1] == train_set.loc[sample]["clase"]:
    #        correct += 1
    #return correct/len(train_set.index)

    # Score prediction accuracy of GLVQ classifier model w_out
    return w_out.score(h_matrix, test_set["clase"])

def model_accuracy(test_train, lmb, n):
    """
    Constructs RVFL prediction models for every study in test_train
    :param test_train: a dictionary mapping the label of each study parent_name to its
    training and testing datasets
    :param lmb: a float value representing the hyperparameter lambda
    :param n: an int value representing the number of neurons in the network's hidden layer
    :return: acc_list: a list of RVFL model accuracies for every study in test_train
    """
    # Instantiate empty list of model accuracies
    acc_list = []
    # Determine constructed model accuracies for datasets across all studies
    for name in test_train:
        train_set = test_train[name]["Train"]
        test_set = test_train[name]["Test"]

        acc = test_train_model(train_set, test_set, lmb, n)
        acc_list.append(acc)
    # Return list of model accuracies
    return acc_list