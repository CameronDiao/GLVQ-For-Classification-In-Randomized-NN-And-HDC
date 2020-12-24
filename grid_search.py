from sklearn.model_selection import ParameterGrid
import scipy.io as sc
import numpy as np
import load_data as ld
import model_accuracy as ma
# Instantiate empty test_train
test_train = {}
# Load test_train with datasets
ld.scan_folder_gs("/Users/camerondiao/Documents/HDResearch/DataManip/data", "data", test_train)

# load optimal hyperparameters
simul=5
#elm_opt_param=sc.loadmat('/Users/camerondiao/Documents/HDResearch/DataManip/i_elm_opt_param.mat') #initial feature set
#elm_opt_param=(elm_opt_param['i_elm_opt_param'])
elm_opt_param=np.genfromtxt('int_lvq_param.csv', delimiter='\t')

# Instantiate list of dataset names
tt_data = sorted(list(test_train.keys()))

# Instantiate ParameterGrid object to perform grid search
grid = {"sigma": [i for i in np.arange(0.6, 1.2, 0.1)]}
param_grid = ParameterGrid(grid)
# Store optimal hyperparameters and corresponding accuracies
best_grid = []
accuracy_all = []

breakpts = [3, 19, 24, 51, 58, 67, 76, 101]

for i in range(102, len(tt_data)): # across all datasets
    if i in breakpts:
        break
    #n = int(elm_opt_param[i, 0])
    #lmb = elm_opt_param[i, 1]
    #kappa = int(elm_opt_param[i, 2])
    ppc = int(elm_opt_param[i, 2])
    beta = int(elm_opt_param[i, 3])

    best_score = 0 # stores best accuracy achieved by hyperparameters
    best_param = [] # stores optimal hyperparameters of dataset

    key = tt_data[i] # dataset name

    for g in param_grid:
        print(i, round(g['sigma'], 1))
        ds_accuracy = [] # stores accuracies obtained by a single set of hyperparameters

        sigma = g['sigma']

        for j in range(simul): # simul tests
            temp = {}
            temp[key] = {}
            temp[key]["Train"] = test_train.get(key)["Train"]
            temp[key]["Test"] = test_train.get(key)["Test"]
            ds_accuracy.append(ma.tt_model_accuracy(temp, None, None, None, ppc, beta, sigma))
            #ds_accuracy.append(ma.tt_model_accuracy(temp, lmb, n, kappa, ppc, beta, sigma))

        accuracy_all_mean = sum(ds_accuracy) / len(ds_accuracy)
        print(accuracy_all_mean)

        # store set of hyperparameters which achieves the highest accuracy so far
        if accuracy_all_mean > best_score:
            best_score = accuracy_all_mean
            best_param = [ppc, beta, round(sigma, 1)]
            #best_param = [n, lmb, kappa, ppc, beta, sigma]

    accuracy_all.append(best_score)
    best_grid.append(best_param)

    accuracy_all_ext = np.array(accuracy_all)
    np.savetxt("s6_11_acc_102.csv", accuracy_all_ext, delimiter='\t')
    best_grid_ext = np.array(best_grid)
    np.savetxt("s6_11_param_102.csv", best_grid_ext, delimiter='\t')

print(sum(accuracy_all) / len(accuracy_all)) # mean accuracy among all datasets

# Output accuracies
#accuracy_all = np.array(accuracy_all)
#np.savetxt("int_lvq_acc.csv", accuracy_all, delimiter='\t')
# Output grid results
#best_grid = np.array(best_grid)
#np.savetxt("int_lvq_param.csv", best_grid, delimiter="\t")

