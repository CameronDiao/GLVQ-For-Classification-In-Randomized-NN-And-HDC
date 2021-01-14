from sklearn.model_selection import ParameterGrid
import scipy.io as sc
import numpy as np
import pandas as pd
import os
import argparse

from model_accuracy import tt_accuracy
from preprocess import scan_folder_gs

def main(hparams):
    test_train = {}
    scan_folder_gs(os.getcwd() + hparams.data_dir, "data", test_train)

    simul=5
    #elm_opt_param=sc.loadmat('/Users/camerondiao/Documents/HDResearch/DataManip/i_elm_opt_param.mat') #initial feature set
    #elm_opt_param=(elm_opt_param['i_elm_opt_param'])
    elm_opt_param=pd.read_csv(os.getcwd() + hparams.param_dir, delimiter='\t')

    tt_data = sorted(list(test_train.keys()))

    grid = {}
    if 'n' in hparams.params:
        grid['n'] = [i for i in range(50, 1550, 50)]
    if 'lmb' in hparams.params:
        grid['lmb'] = [2**i for i in range(-10, 6)]
    if 'kappa' in hparams.params:
        grid['kappa'] = [1, 3, 7, 15]
    if 'epochs' in hparams.params:
        grid['epochs'] = [25, 50, 100, 200]
    if 'ppc' in hparams.params:
        grid['ppc'] = [i for i in range(1, 6)]
    if 'beta' in hparams.params:
        grid['beta'] = [i for i in range(1, 6)]
    if 'sigma' in hparams.params:
        grid['sigma'] = [i for i in np.arange(0.1, 1.2, 0.1)]
    param_grid = ParameterGrid(grid)
    best_grid = []
    accuracy_all = []

    param_types = list(set(grid.keys()) | set(elm_opt_param.columns))

    for i in range(117, len(tt_data)): # across all datasets
        optparams = elm_opt_param.iloc[i].to_dict()

        best_score = 0 # stores best accuracy achieved by hyperparameters
        best_param = [] # stores optimal hyperparameters of dataset

        key = tt_data[i] # dataset name

        for g in param_grid:
            print(i, g)
            ds_accuracy = [] # stores accuracies obtained by a single set of hyperparameters

            optparams.update(g)

            for j in range(simul): # simul tests
                temp = {}
                temp[key] = {}
                temp[key]["Train"] = test_train.get(key)["Train"]
                temp[key]["Test"] = test_train.get(key)["Test"]
                ds_accuracy.append(tt_accuracy(temp, model=hparams.model, classifier=hparams.classifier, optimizer=
                                               hparams.optimizer, **optparams))

            accuracy_all_mean = sum(ds_accuracy) / len(ds_accuracy)
            print(accuracy_all_mean)

            # store set of hyperparameters which achieves the highest accuracy so far
            if accuracy_all_mean > best_score:
                best_score = accuracy_all_mean
                best_param = [optparams[k] for k in param_types]

        accuracy_all.append(best_score)
        best_grid.append(best_param)
        np.savetxt(os.getcwd() + '/b11_15_acc_117.csv', accuracy_all, delimiter='\t')
        save_grid = pd.DataFrame(data=best_grid, columns=param_types)
        save_grid.to_csv(os.getcwd() + '/b11_15_params_117.csv', sep='\t', index=False, header=param_types)


    #np.savetxt(os.getcwd() + '/accuracies.csv', accuracy_all, delimiter='\t')
    #best_grid = pd.DataFrame(data=best_grid, columns=param_types)
    #best_grid.to_csv(os.getcwd() + '/grid_search.csv', sep='\t', index=False, header=param_types)
    print(sum(accuracy_all) / len(accuracy_all)) # mean accuracy among all datasets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GLVQ-RVFL Parameter Tuning')
    parser.add_argument('--params', action='store', nargs = '+', choices=['n', 'lmb', 'kappa', 'ppc', 'beta', 'sigma'],
                        required=True)
    parser.add_argument('--model', action='store', choices=['f', 'c', 'i'], required=True )
    parser.add_argument('--classifier', action='store', required=True)
    parser.add_argument('--optimizer', action='store')
    parser.add_argument('--data_dir', default='/data')
    parser.add_argument('--param_dir', default='/parameters/i_elm_opt_param.csv')
    args = parser.parse_args()
    main(args)

