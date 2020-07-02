from sklearn.model_selection import ParameterGrid
import load_data as ld
import model_accuracy as ma
import numpy as np

# Instantiate empty test_train
test_train = {}
# Load test_train with datasets
ld.scan_folder_gs("/Users/camerondiao/Documents/HDResearch/DataManip/data", "data", test_train)

# load optimal hyperparameters
simul=5
elm_opt_param = np.genfromtxt('final_param.csv', delimiter='\t')

# Gather accuracies for each dataset
tt_data = sorted(list(test_train.keys()))

grid = {"ppc": [i for i in range(1, 6)], "beta": [i for i in range(1, 16)]}
param_grid = ParameterGrid(grid)
best_grid = []
accuracy_all = []

for i in range(len(tt_data)):
    n = int(elm_opt_param[i, 0])
    lmb = elm_opt_param[i, 1]
    kappa = int(elm_opt_param[i, 2])

    best_score = 0
    best_param = {}

    key = tt_data[i]

    for g in param_grid:
        print(i, g)
        ds_accuracy = []

        ppc = g['ppc']
        beta = g['beta']

        for j in range(simul):
            temp = {}
            temp[key] = {}
            temp[key]["Train"] = test_train.get(key)["Train"]
            temp[key]["Test"] = test_train.get(key)["Test"]

            # accuracy_all[i]=tt.model_accuracy(temp, lmb, n)[0]
            ds_accuracy.append(ma.tt_model_accuracy(temp, lmb, n, kappa, ppc, beta))

        accuracy_all_mean = sum(ds_accuracy) / len(ds_accuracy)
        if accuracy_all_mean > best_score:
            best_score = accuracy_all_mean
            best_param = [ppc, beta]

    accuracy_all.append(best_score)
    best_grid.append(best_param)

print(sum(accuracy_all) / len(accuracy_all)) # mean accuracy among all datasets

# Output grid results
best_grid = np.array(best_grid)
np.savetxt("i_elm_opt_param.csv", best_grid, delimiter="\t")

accuracy_all = np.array(accuracy_all)
np.savetxt("accuracy_out.csv", accuracy_all, delimiter="\t")

