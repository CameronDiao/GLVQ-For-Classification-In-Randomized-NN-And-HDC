#from classifiers.sklearnlvq import GlvqModel
#from classifiers.sklearnlvq import RslvqModel
import sklearn_lvq

def lvq1(inputs, labels, classifier, ppc, beta=None, sigma=None):
    if classifier == "glvq":
        w = sklearn_lvq.GlvqModel(prototypes_per_class=ppc, beta=beta)
    elif classifier == "rslvq":
        w = sklearn_lvq.RslvqModel(prototypes_per_class=ppc, sigma=sigma)
    else:
        raise ValueError("Invalid LVQ Classifier Type")
    w.fit(inputs, labels)
    return w