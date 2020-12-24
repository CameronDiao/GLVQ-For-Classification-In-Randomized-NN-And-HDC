import scipy.io as sc
import numpy as np
import pandas as pd
import os
import argparse

from model_accuracy import cv_accuracy, tt_accuracy
from preprocess import scan_folder

def main(hparams):
    test_train = {}
    k_fold = {}
    scan_folder(os.getcwd() + hparams.data_dir, "data", test_train, k_fold)

    simul=5 # number of runs for random parameter initialization
    #elm_opt_param=sc.loadmat('/Users/camerondiao/Documents/HDResearch/DataManip/i_elm_opt_param.mat') #initial feature set
    #elm_opt_param=(elm_opt_param['i_elm_opt_param'])
    elm_opt_param = pd.read_csv(os.getcwd() + hparams.param_dir, delimiter='\t')

    tt_data = list(test_train.keys())
    kf_data = list(k_fold.keys())
    total_data = sorted(tt_data + kf_data)

    accuracy_all = [[] for _ in range(len(total_data))]  # store accuracies for individual datasets

    for sim in range(simul):  # for simul initializations
        for i in range(len(total_data)):
            optparams = elm_opt_param.iloc[i].to_dict()

            print(sim, i)
            key=total_data[i]
            if key in tt_data:  # if dataset is in tt_data
                temp = {}
                temp[key] = {}
                temp[key]["Train"] = test_train.get(key)["Train"]
                temp[key]["Test"] = test_train.get(key)["Test"]

                accuracy_all[i].append(tt_accuracy(temp, model=hparams.model, classifier=hparams.classifier, optimizer=
                                                   hparams.optimizer, **optparams))
            else:  # if dataset is in kf_data
                temp = {}
                temp[key] = {}
                key2 = 1.0
                temp[key][key2] = {}
                temp[key][key2]["Train"] = k_fold.get(key)[key2]["Train"]
                temp[key][key2]["Test"] = k_fold.get(key)[key2]["Test"]
                key2 = 2.0
                temp[key][key2] = {}
                temp[key][key2]["Train"] = k_fold.get(key)[key2]["Train"]
                temp[key][key2]["Test"] = k_fold.get(key)[key2]["Test"]
                key2 = 3.0
                temp[key][key2] = {}
                temp[key][key2]["Train"] = k_fold.get(key)[key2]["Train"]
                temp[key][key2]["Test"] = k_fold.get(key)[key2]["Test"]
                key2 = 4.0
                temp[key][key2] = {}
                temp[key][key2]["Train"] = k_fold.get(key)[key2]["Train"]
                temp[key][key2]["Test"] = k_fold.get(key)[key2]["Test"]

                accuracy_all[i].append(cv_accuracy(temp, model=hparams.model, classifier=hparams.classifier, optimizer=
                                                   hparams.optimizer, **optparams))

    accuracy_all_s = [sum(elm) / simul for elm in accuracy_all]  # mean values for each dataset accross simulations
    np.savetxt(os.getcwd() + '/accuracies.csv', accuracy_all_s, delimiter='\t')

    accuracy_all_mean = sum(accuracy_all_s) / (len(accuracy_all_s) - 8)  # mean accuracy among all datasets
    print(accuracy_all_mean)

    # f = open("output.txt", "w")
    # f.write("Datasets Accuracies:\n")
    # for i in range(len(accuracy_all)):
    #     f.write(total_data[i] + " " + str(accuracy_all[i]) + "\n")
    #
    # f.write("\n")
    # f.write("Average Test/Train Model Accuracy:\n")
    # f.write(str(accuracy_all_mean))
    # f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GLVQ-RVFL')
    parser.add_argument('--model', action='store', choices=['f', 'c', 'i'], required=True )
    parser.add_argument('--classifier', action='store', required=True)
    parser.add_argument('--optimizer', action='store', choices=['lbfgs', 'sgd'])
    parser.add_argument('--data_dir', default='/data')
    parser.add_argument('--param_dir', default='/parameters/int_lvq_param.csv')
    args = parser.parse_args()
    main(args)
