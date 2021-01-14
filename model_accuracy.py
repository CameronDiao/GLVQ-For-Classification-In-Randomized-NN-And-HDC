import re

from models.BaseClassifier import RLMSClassifier, LVQClassifier1, LVQClassifier2
from models.ConvRVFL import ConvRVFLUsingRLMS, ConvRVFLUsingLVQ1, ConvRVFLUsingLVQ2
from models.IntRVFL import IntRVFLUsingRLMS, IntRVFLUsingLVQ1, IntRVFLUsingLVQ2

def cv_accuracy(test_train, **kwargs):
    """
    Computes model accuracy across k folds on the dataset partitioned in test_train
    :param test_train: a dictionary mapping the label of each study parent_name to its dataset,
    indexed by the k folds used to perform CV on the dataset
    :param lmb: a float value representing the hyperparameter lambda
    :param n: an int value representing the number of neurons in the network's hidden layer
    :param kappa: an int value representing the threshold parameter
    :return: acc_list: a list of model accuracies for every study in test_train
    """
    name = list(test_train.keys())[0]
    fold_sets = test_train[name]
    sum_acc = 0
    if kwargs.get('model') == 'f':
        if kwargs.get('classifier') == 'rlms':
            for fold in fold_sets:
                model = RLMSClassifier(fold_sets[fold]["Train"], kwargs.get('lmb'))
                sum_acc += model.score(fold_sets[fold]["Test"])
        elif re.search("^[A-Za-z]+lvq1$", kwargs.get('classifier')):
            for fold in fold_sets:
                model = LVQClassifier1(fold_sets[fold]["Train"], kwargs.get('classifier')[:-1], int(kwargs.get('ppc')),
                                       int(kwargs.get('beta')), kwargs.get('sigma'))
                sum_acc += model.score(fold_sets[fold]["Test"])
        elif re.search("^[A-Za-z]+lvq2$", kwargs.get('classifier')):
            for fold in fold_sets:
                model = LVQClassifier2(fold_sets[fold]["Train"], kwargs.get('classifier')[:-1], kwargs.get('optimizer'),
                                       int(kwargs.get('epochs')), int(kwargs.get('ppc')), int(kwargs.get('beta')),
                                       kwargs.get('sigma'))
                if kwargs.get('optimizer') == 'lbfgs':
                    sum_acc += model.score(fold_sets[fold]["Test"], classifier=kwargs.get('classifier')[:-1], wrapper=True)
                else:
                    sum_acc += model.score(fold_sets[fold]["Test"])
        else:
            raise ValueError("Invalid Classifier Type")
    elif kwargs.get('model') == 'c':
        if kwargs.get('classifier') == 'rlms':
            for fold in fold_sets:
                model = ConvRVFLUsingRLMS(fold_sets[fold]["Train"], int(kwargs.get('n')), kwargs.get('lmb'))
                sum_acc += model.score(fold_sets[fold]["Test"])
        elif re.search("^[A-Za-z]+lvq1$", kwargs.get('classifier')):
            for fold in fold_sets:
                model = ConvRVFLUsingLVQ1(fold_sets[fold]["Train"], kwargs.get('classifier')[:-1], int(kwargs.get('n')),
                                          int(kwargs.get('ppc')), int(kwargs.get('beta')), kwargs.get('sigma'))
                sum_acc += model.score(fold_sets[fold]["Test"])
        elif re.search("^[A-Za-z]+lvq2$", kwargs.get('classifier')):
            for fold in fold_sets:
                model = ConvRVFLUsingLVQ2(fold_sets[fold]["Train"], kwargs.get('classifier')[:-1], kwargs.get('optimizer'),
                                          int(kwargs.get('epochs')), int(kwargs.get('n')), int(kwargs.get('ppc')),
                                          int(kwargs.get('beta')), kwargs.get('sigma'))
                if kwargs.get('optimizer') == 'lbfgs':
                    sum_acc += model.score(fold_sets[fold]["Test"], classifier=kwargs.get('classifier')[:-1], wrapper=True)
                else:
                    sum_acc += model.score(fold_sets[fold]["Test"])
        else:
            raise ValueError("Invalid Classifier Type")
    elif kwargs.get('model') == 'i':
        if kwargs.get('classifier') == 'rlms':
            for fold in fold_sets:
                model = IntRVFLUsingRLMS(fold_sets[fold]["Train"], int(kwargs.get('n')), int(kwargs.get('kappa')),
                                          kwargs.get('lmb'))
                sum_acc += model.score(fold_sets[fold]["Test"])
        elif re.search("^[A-Za-z]+lvq1$", kwargs.get('classifier')):
            for fold in fold_sets:
                model = IntRVFLUsingLVQ1(fold_sets[fold]["Train"], kwargs.get('classifier')[:-1], int(kwargs.get('n')),
                                         int(kwargs.get('kappa')), int(kwargs.get('ppc')), int(kwargs.get('beta')),
                                         kwargs.get('sigma'))
                sum_acc += model.score(fold_sets[fold]["Test"])
        elif re.search("^[A-Za-z]+lvq2$", kwargs.get('classifier')):
            for fold in fold_sets:
                model = IntRVFLUsingLVQ2(fold_sets[fold]["Train"], kwargs.get('classifier')[:-1], kwargs.get('optimizer'),
                                         int(kwargs.get('epochs')), int(kwargs.get('n')), int(kwargs.get('kappa')),
                                         int(kwargs.get('ppc')), int(kwargs.get('beta')), kwargs.get('sigma'))
                if kwargs.get('optimizer') == 'lbfgs':
                    sum_acc += model.score(fold_sets[fold]["Test"], classifier=kwargs.get('classifier')[:-1], wrapper=True)
                else:
                    sum_acc += model.score(fold_sets[fold]["Test"])
        else:
            raise ValueError("Invalid Classifier Type")
    else:
        raise ValueError("Invalid Model Type")
    return sum_acc / len(fold_sets)

def tt_accuracy(test_train, **kwargs):
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

    if kwargs.get('model') == 'f':
        if kwargs.get('classifier') == 'rlms':
            model = RLMSClassifier(train_set, kwargs.get('lmb'))
            return model.score(test_set)
        elif re.search("^[A-Za-z]+lvq1$", kwargs.get('classifier')):
            model = LVQClassifier1(train_set, kwargs.get('classifier')[:-1], int(kwargs.get('ppc')),
                                   int(kwargs.get('beta')), kwargs.get('sigma'))
            return model.score(test_set)
        elif re.search("^[A-Za-z]+lvq2$", kwargs.get('classifier')):
            model = LVQClassifier2(train_set, kwargs.get('classifier')[:-1], kwargs.get('optimizer'),
                                   int(kwargs.get('epochs')), int(kwargs.get('ppc')), int(kwargs.get('beta')),
                                   kwargs.get('sigma'))
            return model.score(test_set, kwargs.get('classifier')[:-1], wrapper=True) \
                if kwargs.get('optimizer') == 'lbfgs' else model.score(test_set)
        else:
            raise ValueError("Invalid Classifier Type")
    elif kwargs.get('model') == 'c':
        if kwargs.get('classifier') == 'rlms':
            model = ConvRVFLUsingRLMS(train_set, kwargs.get('n'), kwargs.get('lmb'))
            return model.score(test_set)
        elif re.search("^[A-Za-z]+lvq1$", kwargs.get('classifier')):
            model = ConvRVFLUsingLVQ1(train_set, kwargs.get('classifier')[:-1], int(kwargs.get('n')),
                                      int(kwargs.get('ppc')), int(kwargs.get('beta')), kwargs.get('sigma'))
            return model.score(test_set)
        elif re.search("^[A-Za-z]+lvq2$", kwargs.get('classifier')):
            model = ConvRVFLUsingLVQ2(train_set, kwargs.get('classifier')[:-1], kwargs.get('optimizer'),
                                      int(kwargs.get('epochs')), int(kwargs.get('n')), int(kwargs.get('ppc')),
                                      int(kwargs.get('beta')), kwargs.get('sigma'))
            return model.score(test_set, kwargs.get('classifier')[:-1], wrapper=True) \
                if kwargs.get('optimizer') == 'lbfgs' else model.score(test_set)
        else:
            raise ValueError("Invalid Classifier Type")
    elif kwargs.get('model') == 'i':
        if kwargs.get('classifier') == 'rlms':
            model = IntRVFLUsingRLMS(train_set, int(kwargs.get('n')), int(kwargs.get('kappa')), kwargs.get('lmb'))
            return model.score(test_set)
        elif re.search("^[A-Za-z]+lvq1$", kwargs.get('classifier')):
            model = IntRVFLUsingLVQ1(train_set, kwargs.get('classifier')[:-1], int(kwargs.get('n')),
                                     int(kwargs.get('kappa')), int(kwargs.get('ppc')), int(kwargs.get('beta')),
                                     kwargs.get('sigma'))
            return model.score(test_set)
        elif re.search("^[A-Za-z]+lvq2$", kwargs.get('classifier')):
            model = IntRVFLUsingLVQ2(train_set, kwargs.get('classifier')[:-1], kwargs.get('optimizer'),
                                     int(kwargs.get('epochs')), int(kwargs.get('n')), int(kwargs.get('kappa')),
                                     int(kwargs.get('ppc')), int(kwargs.get('beta')), kwargs.get('sigma'))
            return model.score(test_set, kwargs.get('classifier')[:-1], wrapper=True) \
                if kwargs.get('optimizer') == 'lbfgs' else model.score(test_set)
        else:
            raise ValueError("Invalid Classifier Type")
    else:
        raise ValueError("Invalid Model Type")
