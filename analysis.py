import numpy as np
import pandas as pd
import math
import load_data as ld
import init
import pred

# Create empty test_train and k_fold dictionaries
test_train = {}
k_fold = {}
# Load test_train and k_fold with their respective datasets
ld.scan_folder("/Users/camerondiao/Documents/HDResearch/DataManip/data", "data", test_train, k_fold)

# Isolate training dataset of the "adult" study
train_set = test_train["adult"]["Train"]
# Redefine index of train_set
train_set.set_index("Unnamed: 0", inplace=True)
# Normalize features in the train_set to the range [0, 1]
train_set = (train_set - train_set.min()) / (train_set.max() - train_set.min())

# Set parameters
lmb = 0.1
n = 100
k = len(train_set.columns) - 1
w_in = np.random.uniform(-1, 1, size=(n, k))
b = np.random.uniform(-0.1, 0.1, size=(n, 1))
g = lambda num: 1/(1 + math.exp(-num))

# Compute the activation matrix of the hidden layer
train_features = train_set.drop(["clase"], axis=1)
h_matrix = init.activation_matrix(train_features, w_in, b, g)
# Store ground truth classifications of training examples using one-hot encodings
y_matrix = pd.get_dummies(train_set["clase"])
# Compute the readout matrix W_out
w_out = init.readout_matrix(h_matrix, y_matrix, lmb, n)

# Isolate testing dataset of the "adult" study
test_set = test_train["adult"]["Test"]
# Redefine index of test_set
test_set.set_index("Unnamed: 0", inplace = True)
# Normalize features in the test_set to the range [0, 1]
test_set = (test_set - test_set.min()) / (test_set.max() - test_set.min())

# Compute the activation matrix of the hidden layer
test_features = test_set.drop(["clase"], axis = 1)
h_matrix = init.activation_matrix(test_features, w_in, b, g)
# Compute the dot product between h_matrix and w_out
test_pred = h_matrix.dot(w_out)

# Generate test_set predictions
pred_series = pred.generate_pred(test_pred)

# test accuracy
correct = 0
for sample in test_set.index:
    if pred_series[sample] == test_set.loc[sample]["clase"]:
        correct += 1
print(correct, len(test_set.index), correct/len(test_set.index))
