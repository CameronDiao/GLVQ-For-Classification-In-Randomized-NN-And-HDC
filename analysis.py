import sys
import load_data as ld
import test_train as tt
import k_fold as kf

# Instantiate empty test_train and k_fold dictionaries
test_train = {}
k_fold = {}
# Load test_train and k_fold with their respective datasets
ld.scan_folder("/Users/camerondiao/Documents/HDResearch/DataManip/data", "data", test_train, k_fold)

# Set parameters
lmb = 0.1
n = 100

# Gather model accuracies for each dataset
tt_list = tt.model_accuracy(test_train, lmb, n)
kf_list = kf.model_accuracy(k_fold, lmb, n)

# Find average model accuracy for the training/testing datasets
avg_tt_acc = 0
for acc in tt_list:
    avg_tt_acc += acc
avg_tt_acc = avg_tt_acc/len(tt_list)

# Find average model accuracy for the cross validation datasets
avg_kf_acc = 0
for acc in kf_list:
    avg_kf_acc += acc
avg_kf_acc = avg_kf_acc/len(kf_list)

# Output results
sys.stdout = open("output.txt", "w")

print("Test/Train Model Accuracies: ", tt_list)
print("Cross Validation Model Accuracies: ", kf_list)
print("Average Test/Train Model Accuracy: ", avg_tt_acc)
print("Average Cross Validation Model Accuracy: ", avg_kf_acc)

sys.stdout.close()

