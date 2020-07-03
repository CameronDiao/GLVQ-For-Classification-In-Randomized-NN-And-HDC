import test_train as tt

def kf_model_accuracy(test_train, lmb, n, kappa, ppc, beta):
    """
    Computes model accuracy across k folds on the dataset partitioned in test_train
    :param test_train: a dictionary mapping the label of each study parent_name to its dataset,
    indexed by the k folds used to perform CV on the dataset
    :param lmb: a float value representing the hyperparameter lambda
    :param n: an int value representing the number of neurons in the network's hidden layer
    :param kappa: an int value representing the threshold parameter
    :return: acc_list: a list of model accuracies for every study in test_train
    """
    # Isolate dataset
    name = list(test_train.keys())[0]
    fold_sets = test_train[name]
    # Count number of folds
    num_folds = len(fold_sets)
    sum_acc = 0
    # Calculate model accuracy for individual folds
    for fold in fold_sets:
        # For RVFL networks
        #fold_acc = tt.encoding_model(fold_sets[fold]["Train"], fold_sets[fold]["Test"], lmb, n, kappa)
        # For LVQ networks
        fold_acc = tt.lvq_model(fold_sets[fold]["Train"], fold_sets[fold]["Test"], n, kappa, ppc, beta)
        # For LVQ classifiers
        #fold_acc = tt.direct_lvq_model(fold_sets[fold]["Train"], fold_sets[fold]["Test"], ppc, beta)
        sum_acc += fold_acc
        #iter_acc.update(temp_acc)
    # Return average model accuracy across all folds
    return sum_acc / num_folds

def tt_model_accuracy(test_train, lmb, n, kappa, ppc, beta):
    """
    Computes model accuracy on the dataset in test_train
    :param test_train: a dictionary mapping the label of each study parent_name to its
    training and testing datasets
    :param lmb: a float value representing the hyperparameter lambda
    :param n: an int value representing the number of neurons in the network's hidden layer
    :param kappa: an int value representing the threshold parameter
    :return: acc_list: a list of model accuracies for every study in test_train
    """
    name = list(test_train.keys())[0]
    train_set = test_train[name]["Train"]
    test_set = test_train[name]["Test"]

    # For RVFL networks
    # acc = tt.encoding_model(train_set, test_set, lmb, n, kappa)
    # For LVQ networks
    acc = tt.lvq_model(train_set, test_set, n, kappa, ppc, beta)
    # For LVQ classifiers
    # acc = tt.direct_lvq_model(train_set, test_set, ppc, beta)
    return acc
