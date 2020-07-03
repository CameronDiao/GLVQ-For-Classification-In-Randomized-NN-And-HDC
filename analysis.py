import time
import scipy.io as sc
import numpy as np
import load_data as ld
import model_accuracy as ma

# Instantiate empty test_train and k_fold dictionaries
test_train = {}
k_fold = {}
# Load test_train and k_fold with their respective datasets
ld.scan_folder("/Users/camerondiao/Documents/HDResearch/DataManip/data", "data", test_train, k_fold)

# load optimal hyperparameters
simul=5 # number of runs for random parameter initialization
#elm_opt_param=sc.loadmat('/Users/camerondiao/Documents/HDResearch/DataManip/i_elm_opt_param.mat') #initial feature set
#elm_opt_param=(elm_opt_param['i_elm_opt_param'])
elm_opt_param = np.genfromtxt('final_param.csv', delimiter='\t')

# Gather accuracies for each dataset
tt_data = list(test_train.keys())
kf_data = list(k_fold.keys())
total_data = sorted(tt_data + kf_data)

accuracy_all = [[] for i in range(len(total_data))]  # store accuracies for individual datasets

start_time = time.time()

for sim in range(simul):  # for simul initializations
    for i in range(len(total_data)):
        n = int(elm_opt_param[i, 0])
        lmb = elm_opt_param[i, 1]
        kappa = int(elm_opt_param[i, 2])

        ppc = int(elm_opt_param[i, 3])
        beta = int(elm_opt_param[i, 4])

        print(sim, i)
        key=total_data[i]
        if key in tt_data:  # if dataset is in tt_data
            # Recreate dictionary structure with only the current dataset
            temp = {}
            temp[key] = {}
            temp[key]["Train"] = test_train.get(key)["Train"]
            temp[key]["Test"] = test_train.get(key)["Test"]

            accuracy_all[i].append(ma.tt_model_accuracy(temp, n, lmb, kappa, ppc, beta))
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

            # Recreate dictionary structure with only the current dataset
            accuracy_all[i].append(ma.kf_model_accuracy(temp, n, lmb, kappa, ppc, beta))

# Accuracy mean
accuracy_all_s = [sum(elm) / simul for elm in accuracy_all]  # mean values for each dataset accross simulations

accuracy_all_mean = sum(accuracy_all_s) / len(accuracy_all_s)  # mean accuracy among all datasets

print("--- %s seconds ---" % (time.time() - start_time))

# Output results
f = open("output.txt", "w")
f.write("Datasets Accuracies:\n")
for i in range(len(accuracy_all)):
    f.write(total_data[i] + " " + str(accuracy_all[i]) + "\n")

f.write("\n")
f.write("Average Test/Train Model Accuracy:\n")
f.write(str(accuracy_all_mean))
f.close()
