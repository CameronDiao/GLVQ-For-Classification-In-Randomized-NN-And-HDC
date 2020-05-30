import sys
import load_data as ld
import test_train as tt
import k_fold as kf
#import scipy.io as sc

# Instantiate empty test_train and k_fold dictionaries
test_train = {}
k_fold = {}
# Load test_train and k_fold with their respective datasets
ld.scan_folder("/Users/camerondiao/Documents/HDResearch/DataManip/data", "data", test_train, k_fold)
# ld.scan_folder("/Users/denkle/Dropbox/Google_Drive/WORK/MATLAB/2020_intRVFL_centroids_vs_readout/many_datasets/data",
#               "data", test_train, k_fold)

# Set parameters
lmb = 0.1
n = 100

# load optimal hyperparameters
simul = 5  # number of runs for random parameter initialization
# elm_opt_param = sc.loadmat('/Users/denkle/Dropbox/Google_Drive/WORK/MATLAB/2017_ELM/elm_opt_param.mat')  # initial feature set
# elm_opt_param = (elm_opt_param['elm_opt_param'])

# Gather accuracies for each dataset
tt_data = list(test_train.keys())
kf_data = list(k_fold.keys())
total_data = sorted(tt_data + kf_data)

# accuracy_all=[0]*len(total_data)
accuracy_all = [[] for i in range(len(total_data))]  # will store accuracies for individual datasets

for sim in range(simul):  # for simul initializations
    for i in range(len(total_data)):
        # n = int(elm_opt_param[i, 0])
        # lmb = elm_opt_param[i, 1]

        # print(sim, i)
        key=total_data[i]
        if key in tt_data:  # if dataset is in tt_data
            # Recreate dictionary structure with only the current dataset
            temp = {}
            temp[key] = {}
            temp[key]["Train"] = test_train.get(key)["Train"]
            temp[key]["Test"] = test_train.get(key)["Test"]

            # accuracy_all[i]=tt.model_accuracy(temp, lmb, n)[0]
            accuracy_all[i].append(tt.model_accuracy(temp, lmb, n)[0])
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

            # Recteate dictionary structure with only the current dataset
            # accuracy_all[i]=kf.model_accuracy(temp, lmb, n)[0]
            accuracy_all[i].append(kf.model_accuracy(temp, lmb, n)[0])

# Accuracy mean
accuracy_all_s = [sum(elm) / simul for elm in accuracy_all]  # mean values for each dataset accross simulations

accuracy_all_mean = sum(accuracy_all_s) / len(accuracy_all_s)  # mean accuracy among all datasets

# Output results
f = open("output.txt", "w")
f.write("Datasets Accuracies:\n")
for i in range(len(accuracy_all)):
    f.write(total_data[i] + " " + str(accuracy_all[i]) + "\n")

f.write("\n")
f.write("Average Test/Train Model Accuracy:\n")
f.write(str(accuracy_all_mean))
f.close()

# Gather accuracies for each dataset
# tt_list = tt.model_accuracy(test_train, lmb, n)
# kf_list = kf.model_accuracy(k_fold, lmb, n)


# Find average model accuracy for the training/testing datasets
# avg_tt_acc = 0
# for acc in tt_list:
#    avg_tt_acc += acc
# avg_tt_acc = avg_tt_acc/len(tt_list)

# Find average model accuracy for the cross validation datasets
# avg_kf_acc = 0
# for acc in kf_list:
#    avg_kf_acc += acc
# avg_kf_acc = avg_kf_acc/len(kf_list)


# Output results


# print("Datasets Accuracies:")
# print(accuracy_all)
#
# print("Average Test/Train Model Accuracy:")
# print(accuracy_all_mean)
#
##print(tt_list)
##print(kf_list)
##print(avg_tt_acc)
##print(avg_kf_acc)
#


# Output results
# sys.stdout = open("output.txt", "w")
# print("Datasets Accuracies:")
# print(accuracy_all)
#
# print("Average Test/Train Model Accuracy:")
# print(accuracy_all_mean)
#
##print(tt_list)
##print(kf_list)
##print(avg_tt_acc)
##print(avg_kf_acc)
#
# sys.stdout.close()
