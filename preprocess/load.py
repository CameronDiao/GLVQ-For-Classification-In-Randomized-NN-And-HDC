from preprocess.read import read_text, read_cv, read_tt
from preprocess.process import normalize
import pandas as pd
import os
import re

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
            train_reader = read_text(train_set)
            train_reader = normalize(train_reader)

            test_reader = read_text(test_set)
            test_reader = normalize(test_reader)

            data_list = {"Train": train_reader, "Test": test_reader}
            test_train[parent_name] = data_list
        else:
            data_set = "".join((parent, "/", test_type))
            data_reader = read_text(data_set)
            data_reader = normalize(data_reader)

            idx_set = "".join((parent, "/conxuntos_kfold.dat"))
            idx_reader = read_cv(idx_set, data_reader)
            k_fold[parent_name] = idx_reader

    else:
        for file_name in os.listdir(parent):
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                scan_folder(current_path, file_name, test_train, k_fold)

def scan_folder_gs(parent, parent_name, test_train):
    """
    Separates the files of a given folder into testing/training datasets for parameter tuning
    :param parent: the file path of the given folder/file
    :param parent_name: the name of the folder/file
    :param test_train: a dictionary mapping the label of each study parent_name to its dataset
    :return: None
    """
    files_in = [item for item in os.listdir(parent) if re.match(".+\..+", item)]
    if len(files_in) > 0:
        test_type = parent_name + "_R.dat"
        if test_type not in files_in:
            data_set = "".join((parent, "/", parent_name + "_train_R.dat"))
            data_reader = read_text(data_set)
            data_reader = normalize(data_reader)

            idx_set = "".join((parent, "/conxuntos.dat"))
            idx_reader = read_tt(idx_set, data_reader)
            test_train[parent_name] = idx_reader
        else:
            data_set = "".join((parent, "/", test_type))
            data_reader = read_text(data_set)
            data_reader = normalize(data_reader)

            idx_set = "".join((parent, "/conxuntos.dat"))
            idx_reader = read_tt(idx_set, data_reader)
            test_train[parent_name] = idx_reader
    else:
        for file_name in os.listdir(parent):
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                scan_folder_gs(current_path, file_name, test_train)