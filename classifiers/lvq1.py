from classifiers.sklearnlvq import GlvqModel
from classifiers.sklearnlvq import RslvqModel
#import sklearn_lvq

def lvq1(inputs, labels, classifier, epochs, ppc, beta=None, sigma=None):
    if classifier == "glvq":
        w = GlvqModel(prototypes_per_class=ppc, beta=beta, max_iter=epochs)
    elif classifier == "rslvq":
        w = RslvqModel(prototypes_per_class=ppc, sigma=sigma, max_iter=epochs)
    else:
        raise ValueError("Invalid LVQ Classifier Type")
    w.fit(inputs, labels)
    return w