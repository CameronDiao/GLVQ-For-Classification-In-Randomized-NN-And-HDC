import pandas as pd
import os
import re
import init

def read_set(file_path):
    """
    Reads a tab-separated text file into a pandas DataFrame
    :param file_path: the file path of the tab-separated file
    :return: reader: a DataFrame object containing the contents of the file
    """
    with open(file_path, "r") as f:
        reader = pd.read_csv(f, delimiter = "\t")
    return reader

def read_indices(file_path, dataset):
    """
    Assumes k-fold cross validation was performed on a given dataset
    Returns a dictionary mapping the ith fold to its corresponding testing and training datasets
    :param file_path: the file name of the tab-separated file containing the k-fold indices
    :param dataset: the entire dataset
    :return: fold_sets: a dictionary mapping the ith fold to a dictionary containing the testing set and
    the training set
    """
    fold_sets = {}
    with open(file_path, "r") as f:
        reader = f.readlines()
    for idx in range(0, len(reader), 2):
        train_indices = reader[idx].strip().split(" ")
        test_indices = reader[idx + 1].strip().split(" ")
        train_set = dataset[dataset.index.isin(train_indices)]
        test_set = dataset[dataset.index.isin(test_indices)]
        idx_list = {"Train": train_set, "Test": test_set}
        fold_sets[(idx / 2) + 1] = idx_list
    return fold_sets

def read_test_train(file_path, dataset):
    with open(file_path, "r") as f:
        reader = f.readlines()
    train_indices = reader[0].strip().split(" ")
    test_indices = reader[1].strip().split(" ")
    train_set = dataset[dataset.index.isin(train_indices)]
    test_set = dataset[dataset.index.isin(test_indices)]
    tt_sets = {"Train": train_set, "Test": test_set}
    return tt_sets

def scan_folder(parent, parent_name, test_train, k_fold):
    """
    Separates the files of a given folder into testing/training datasets and cross validation datasets
    :param parent: the file path of the given folder/file
    :param parent_name: the name of the folder/file
    :param test_train: a dictionary mapping the label of each study parent_name to its dataset
    :param k_fold: a dictionary mapping the label of each study parent_name to its dataset,
    indexed by the k folds used to perform cross validation on the dataset
    :return: None
    """
    files_in = [item for item in os.listdir(parent) if re.match(".+\..+", item)]
    if len(files_in) > 0:
        test_type = parent_name + "_R.dat"
        if test_type not in files_in:
            train_set = "".join((parent, "/", parent_name + "_train_R.dat"))
            test_set = "".join((parent, "/", parent_name + "_test_R.dat"))
            train_reader = read_set(train_set)
            train_reader = init.preprocess(train_reader)

            test_reader = read_set(test_set)
            test_reader = init.preprocess(test_reader)

            data_list = {"Train": train_reader, "Test": test_reader}
            test_train[parent_name] = data_list
        else:
            data_set = "".join((parent, "/", test_type))
            data_reader = read_set(data_set)
            data_reader = init.preprocess(data_reader)

            idx_set = "".join((parent, "/conxuntos_kfold.dat"))
            idx_reader = read_indices(idx_set, data_reader)
            k_fold[parent_name] = idx_reader

    else:
        for file_name in os.listdir(parent):
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                # if we're checking a sub-directory, recall this method
                scan_folder(current_path, file_name, test_train, k_fold)

def scan_folder_gs(parent, parent_name, test_train):
    files_in = [item for item in os.listdir(parent) if re.match(".+\..+", item)]
    if len(files_in) > 0:
        test_type = parent_name + "_R.dat"
        if test_type not in files_in:
            train_set = "".join((parent, "/", parent_name + "_train_R.dat"))
            test_set = "".join((parent, "/", parent_name + "_test_R.dat"))
            train_reader = read_set(train_set)
            train_reader = init.preprocess(train_reader)

            test_reader = read_set(test_set)
            test_reader = init.preprocess(test_reader)

            data_list = {"Train": train_reader, "Test": test_reader}
            test_train[parent_name] = data_list
        else:
            data_set = "".join((parent, "/", test_type))
            data_reader = read_set(data_set)
            data_reader = init.preprocess(data_reader)

            idx_set = "".join((parent, "/conxuntos.dat"))
            idx_reader = read_test_train(idx_set, data_reader)
            test_train[parent_name] = idx_reader
    else:
        for file_name in os.listdir(parent):
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                # if we're checking a sub-directory, recall this method
                scan_folder_gs(current_path, file_name, test_train)
