import test_train as tt

def kf_model_accuracy(test_train, lmb, n, kappa):
    """
    Computes prediction model accuracy across k folds on the dataset partitioned in test_train
    :param test_train: a dictionary mapping the label of each study parent_name to its dataset,
    indexed by the k folds used to perform cross validation on the dataset
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
    #iter_acc = {}
    for fold in fold_sets:
        # Calculate model accuracy for individual folds
        fold_acc = tt.lvq_model(fold_sets[fold]["Train"], fold_sets[fold]["Test"], n, kappa) #temp_acc
        sum_acc += fold_acc
        #iter_acc.update(temp_acc)
    # Return average model accuracy across all folds
    return sum_acc / num_folds #iter_acc

def tt_model_accuracy(test_train, lmb, n, kappa):
    """
    Computes prediction model accuracy on the dataset in test_train
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

    acc = tt.lvq_model(train_set, test_set, n, kappa) #iter_acc
    return acc #iter_acc
