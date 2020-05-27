import numpy as np
import pandas as pd
import os
import re

def read_set(file_path):
    """
    Reads a tab-separated text file into a pandas DataFrame
    :param file_path: the file path of the tab-separated file
    :return: reader: a pandas DataFrame
    """
    with open(file_path, "r") as f:
        reader = pd.read_csv(f, delimiter = "\t")
    return reader

def read_indices(file_path):
    """
    Assumes k-fold cross validation was performed on a given dataset
    Returns a dictionary mapping the ith fold to the testing and training indices of that dataset
    :param file_path: the file name of the tab-separated file containing the dataset
    :return: fold_indices: a dictionary mapping the ith fold to a dictionary containing the testing indices and
    the training indices
    """
    fold_indices = {}
    with open(file_path, "r") as f:
        reader = f.readlines()
    for idx in range(0, len(reader), 2):
        idx_list = {"Train": reader[idx].strip().split(" "), "Test": reader[idx + 1].strip().split(" ")}
        fold_indices[(idx / 2) + 1] = idx_list
    return fold_indices

def scan_folder(parent, parent_name, test_train, k_fold):
    """
    Separates the files of a given folder into testing/training datasets and cross validation datasets
    :param parent: the file path of the given folder/file
    :param parent_name: the name of the folder/file
    :param test_train: a dictionary mapping the label of each dataset parent_name to the dataset itself
    :param k_fold: a dictionary mapping the label of each dataset parent_name to the dataset itself,
    with the dataset's cross validation indices
    :return: None
    """
    files_in = [item for item in os.listdir(parent) if re.match(".+\..+", item)]
    if len(files_in) > 0:
        test_type = parent_name + "_R.dat"
        if test_type not in files_in:
            train_set = "".join((parent, "/", parent_name + "_train_R.dat"))
            test_set = "".join((parent, "/", parent_name + "_test_R.dat"))
            train_reader = read_set(train_set)
            test_reader = read_set(test_set)
            data_list = {"Train": train_reader, "Test": test_reader}
            test_train[parent_name] = data_list
        else:
            data_set = "".join((parent, "/", test_type))
            data_reader = read_set(data_set)
            idx_set = "".join((parent, "/conxuntos_kfold.dat"))
            idx_reader = read_indices(idx_set)
            k_fold[parent_name] = {"Data": data_reader, "Indices": idx_reader}

    else:
        for file_name in os.listdir(parent):
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                # if we're checking a sub-directory, recall this method
                scan_folder(current_path, file_name, test_train, k_fold)