import numpy as np
import pandas as pd
import os
import csv
import re


def scan_folder(parent, parent_name, test_train, k_fold):
    files_in = [item for item in os.listdir(parent) if "." in item]
    if len(files_in) > 0 and files_in != ['.DS_Store']:
        test_type = parent_name + "_R.dat"
        if test_type not in files_in:
            test_set = "".join((parent, "/", parent_name + "_test_R.dat"))
            train_set = "".join((parent, "/", parent_name + "_train_R.dat"))
            with open(test_set, 'r') as f:
                test_reader = pd.read_csv(f, delimiter = "\t")
            with open(train_set, 'r') as f:
                train_reader = pd.read_csv(f, delimiter = "\t")
            data_list = [test_reader, train_reader]
            test_train[parent_name] = data_list
        else:
            test_type = "".join((parent, "/", test_type))
            with open(test_type, 'r') as f:
                reader = pd.read_csv(f, delimiter = "\t")
                k_fold[parent_name] = reader
    else:
        for file_name in os.listdir(parent):
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                # if we're checking a sub-directory, recall this method
                scan_folder(current_path, file_name, test_train, k_fold)


test_train = {}
k_fold = {}
scan_folder("/Users/camerondiao/Documents/HDResearch/DataManip/data", "data", test_train, k_fold)
