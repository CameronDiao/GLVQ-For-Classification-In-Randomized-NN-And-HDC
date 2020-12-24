import time
import load_data as ld
import model_accuracy as ma
import scipy.io as sc
import numpy as np

# Instantiate empty test_train and k_fold dictionaries
test_train = {}
k_fold = {}
# Load test_train and k_fold with their respective datasets
ld.scan_folder("/Users/camerondiao/Documents/HDResearch/DataManip/data", "data", test_train, k_fold)

# load optimal hyperparameters
simul=5 # number of runs for random parameter initialization
elm_opt_param = np.genfromtxt('final_param.csv', delimiter='\t')

accuracy_all = []

start_time = time.time()

for sim in range(simul):  # for simul initializations
    n = int(elm_opt_param[0, 0])
    lmb = elm_opt_param[0, 1]
    kappa = int(elm_opt_param[0, 2])

    ppc = int(elm_opt_param[0, 3])
    beta = int(elm_opt_param[0, 4])

    print(sim, 0)

    temp = {}
    temp["abalone"] = {}
    sub = 1.0
    temp["abalone"][sub] = {}
    temp["abalone"][sub]["Train"] = k_fold.get("abalone")[sub]["Train"]
    temp["abalone"][sub]["Test"] = k_fold.get("abalone")[sub]["Test"]
    sub = 2.0
    temp["abalone"][sub] = {}
    temp["abalone"][sub]["Train"] = k_fold.get("abalone")[sub]["Train"]
    temp["abalone"][sub]["Test"] = k_fold.get("abalone")[sub]["Test"]
    sub = 3.0
    temp["abalone"][sub] = {}
    temp["abalone"][sub]["Train"] = k_fold.get("abalone")[sub]["Train"]
    temp["abalone"][sub]["Test"] = k_fold.get("abalone")[sub]["Test"]
    sub = 4.0
    temp["abalone"][sub] = {}
    temp["abalone"][sub]["Train"] = k_fold.get("abalone")[sub]["Train"]
    temp["abalone"][sub]["Test"] = k_fold.get("abalone")[sub]["Test"]

    set_acc = ma.kf_model_accuracy(temp, lmb, n, kappa, ppc, beta, sigma=0.5)
    print(set_acc)
    accuracy_all.append(set_acc)

#    for k,v in set_acc[1].items():
#        iter_acc.setdefault(k, []).append(v)

#for k,v in iter_acc.items():
#    iter_acc[k] = sum(v) / len(v)

print(accuracy_all)
# Accuracy mean
accuracy_all_mean = sum(accuracy_all) / len(accuracy_all)  # mean accuracy among all datasets
print(accuracy_all_mean)
print("--- %s seconds ---" % (time.time() - start_time))

# Output results
#f = open("output.txt", "w")
#f.write("Datasets Accuracies:\n")
#for i in range(len(accuracy_all)):
#    f.write("abalone" + " " + str(accuracy_all[i]) + "\n")

#f.write("\n")
#f.write("Average Test/Train Model Accuracy:\n")
#f.write(str(accuracy_all_mean))
#f.close()

