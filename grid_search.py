from sklearn.model_selection import ParameterGrid
import numpy as np
import load_data as ld
import model_accuracy as ma

# Instantiate empty test_train
test_train = {}
# Load test_train with datasets
ld.scan_folder_gs("/Users/camerondiao/Documents/HDResearch/DataManip/data", "data", test_train)

# load optimal hyperparameters
simul=5
elm_opt_param = np.genfromtxt('final_param.csv', delimiter='\t')

# Instantiate list of dataset names
tt_data = sorted(list(test_train.keys()))

# Instantiate ParameterGrid object to perform grid search
grid = {"beta": [i for i in range(1, 16)]}
param_grid = ParameterGrid(grid)
# Store optimal hyperparameters and corresponding accuracies
best_grid = []
accuracy_all = []

for i in range(len(tt_data)): # across all datasets
    n = int(elm_opt_param[i, 0])
    lmb = elm_opt_param[i, 1]
    kappa = int(elm_opt_param[i, 2])

    best_score = 0 # stores best accuracy achieved by hyperparameters
    best_param = {} # stores optimal hyperparameters of dataset

    key = tt_data[i] # dataset name

    for g in param_grid:
        print(i, g)
        ds_accuracy = [] # stores accuracies obtained by a single set of hyperparameters

        ppc = 1
        beta = g['beta']

        for j in range(simul): # simul tests
            temp = {}
            temp[key] = {}
            temp[key]["Train"] = test_train.get(key)["Train"]
            temp[key]["Test"] = test_train.get(key)["Test"]

            ds_accuracy.append(ma.tt_model_accuracy(temp, lmb, n, kappa, ppc, beta))

        accuracy_all_mean = sum(ds_accuracy) / len(ds_accuracy)

        # store set of hyperparameters which achieves the highest accuracy so far
        if accuracy_all_mean > best_score:
            best_score = accuracy_all_mean
            best_param = [n, lmb, kappa, ppc, beta]

    accuracy_all.append(best_score)
    best_grid.append(best_param)

print(sum(accuracy_all) / len(accuracy_all)) # mean accuracy among all datasets

# Output grid results
best_grid = np.array(best_grid)
np.savetxt("ppc_1_de_model.csv", best_grid, delimiter="\t")

