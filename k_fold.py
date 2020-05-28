import test_train as tt

def k_fold_cv(fold_sets, lmb, n):
    """
    Computes the average model accuracy across k folds of the dataset partitioned in fold_sets
    :param fold_sets: a dictionary mapping the number i to the ith fold's training and testing datasets
    :param lmb: a float value representing the hyperparameter lambda
    :param n: an int value representing the number of neurons in the network's hidden layer
    :return: the average model accuracy across k folds of the dataset partitioned in fold_sets
    """
    # Count number of folds
    num_folds = len(fold_sets)
    sum_acc = 0
    for fold in fold_sets:
        # Calculate model accuracy for individual folds
        fold_acc = tt.test_train_model(fold_sets[fold]["Train"], fold_sets[fold]["Test"], lmb, n)
        sum_acc += fold_acc
    # Return average model accuracy across all folds
    return sum_acc / (num_folds)

def model_accuracy(test_train, lmb, n):
    """
    Constructs RVFL prediction models for every study in test_train
    :param test_train: a dictionary mapping the label of each study parent_name to its dataset,
    indexed by the k folds used to perform cross validation on the dataset
    :param lmb: a float value representing the hyperparameter lambda
    :param n: an int value representing the number of neurons in the network's hidden layer
    :return: acc_list: a list of RVFL model accuracies for every study in test_train
    """
    # Instantiate empty list of model accuracies
    acc_list = []
    # Determine constructed model accuracies for datasets across all studies
    for name in test_train:
        acc = k_fold_cv(test_train[name], lmb, n)
        acc_list.append(acc)
    # Return list of model accuracies
    return acc_list