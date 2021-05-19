import pandas as pd

def read_text(file_path):
    """
    Reads a tab-separated text file into a pandas DataFrame
    :param file_path: the file path of the tab-separated file
    :return: reader: a DataFrame object containing the contents of the file
    """
    with open(file_path, "r") as f:
        reader = pd.read_csv(f, delimiter = "\t")
    return reader

def read_cv(file_path, dataset):
    """
    Assumes k-fold cross validation was performed on a given dataset
    Returns a dictionary mapping the ith fold to its corresponding testing and training datasets
    :param file_path: the file path of the tab-separated file containing the k-fold indices
    :param dataset: the entire dataset
    :return: cv_sets: a dictionary mapping the ith fold to a dictionary containing the testing set and
    the training set
    """
    cv_sets = {}
    with open(file_path, "r") as f:
        reader = f.readlines()
    for idx in range(0, len(reader), 2):
        train_indices = reader[idx].strip().split(" ")
        test_indices = reader[idx + 1].strip().split(" ")
        train_set = dataset[dataset.index.isin(train_indices)]
        test_set = dataset[dataset.index.isin(test_indices)]
        idx_list = {"Train": train_set, "Test": test_set}
        cv_sets[(idx / 2) + 1] = idx_list
    return cv_sets

def read_tt(file_path, dataset):
    """
    Return a dictionary containing the testing set and the training set
    :param file_path: the file path of the tab-separated file
    :param dataset: the entire dataset
    :return: tt_sets: a dictionary containing the testing set and the training set
    """
    with open(file_path, "r") as f:
        reader = f.readlines()
    train_indices = reader[0].strip().split(" ")
    test_indices = reader[1].strip().split(" ")
    train_set = dataset[dataset.index.isin(train_indices)]
    test_set = dataset[dataset.index.isin(test_indices)]
    tt_sets = {"Train": train_set, "Test": test_set}
    return tt_sets