import numpy as np
import pandas as pd
import math
import load_data as ld
import training

# Create empty test_train and k_fold dictionaries
test_train = {}
k_fold = {}
# Load test_train and k_fold with their respective datasets
ld.scan_folder("/Users/camerondiao/Documents/HDResearch/DataManip/data", "data", test_train, k_fold)

# Isolate training dataset of the "adult" study
sample_set = test_train["adult"]["Train"]
# Redefine index of sample_set
sample_set.set_index("Unnamed: 0", inplace=True)
# Normalize features in the sample_set to the range [0, 1]
sample_set=(sample_set-sample_set.min())/(sample_set.max() - sample_set.min())

# Set parameters
lmb = 0.1
k = len(sample_set.columns)
n = 100
w_in = np.random.uniform(-1, 1, size=(n, k))
b = np.random.uniform(-0.1, 0.1, size=(n, 1))
g = lambda num: 1/(1 + math.exp(-num))

# Compute the activation matrix of the hidden layer
h_matrix = training.activation_matrix(sample_set, w_in, b, g)

# Store ground truth classifications of training examples using one-hot encodings
y_matrix = pd.get_dummies(sample_set["clase"])

# Compute the readout matrix W_out
w_out = training.readout_matrix(h_matrix, y_matrix, lmb, n)
w_out = w_out.transpose()

