# -*- coding: utf-8 -*-

# Author: Joris Jensen <jjensen@techfak.uni-bielefeld.de>
#
# License: BSD 3 clause

from __future__ import division

import numpy as np
import numexpr as ne
from numba import njit
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from sklearn.utils import validation
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from itertools import product

from lvq import _LvqBaseModel

def _squared_euclidean(a, b=None):
    if b is None:
        sub = a.dot(a.T)
        d = ne.evaluate("sum(a ** 2, 1)")[np.newaxis].T + ne.evaluate("sum(a ** 2, 1)") - \
            ne.evaluate("2 * sub")
    else:
        sub = a.dot(b.T)
        d = ne.evaluate("sum(a ** 2, 1)")[np.newaxis].T + ne.evaluate("sum(b ** 2, 1)") - \
            ne.evaluate("2 * sub")
    return np.maximum(d, 0)

class GlvqModel(_LvqBaseModel):
    """Generalized Learning Vector Quantization
    Parameters
    ----------
    prototypes_per_class : int or list of int, optional (default=1)
        Number of prototypes per class. Use list to specify different
        numbers per class.
    initial_prototypes : array-like, shape =  [n_prototypes, n_features + 1],
     optional
        Prototypes to start with. If not given initialization near the class
        means. Class label must be placed as last entry of each prototype.
    max_iter : int, optional (default=2500)
        The maximum number of iterations.
    gtol : float, optional (default=1e-5)
        Gradient norm must be less than gtol before successful termination
        of bfgs.
    beta : int, optional (default=2)
        Used inside phi.
        1 / (1 + np.math.exp(-beta * x))
    C : array-like, shape = [2,3] ,optional
        Weights for wrong classification of form (y_real,y_pred,weight)
        Per default all weights are one, meaning you only need to specify
        the weights not equal one.
    display : boolean, optional (default=False)
        Print information about the bfgs steps.
    random_state : int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Attributes
    ----------
    w_ : array-like, shape = [n_prototypes, n_features]
        Prototype vector, where n_prototypes in the number of prototypes and
        n_features is the number of features
    c_w_ : array-like, shape = [n_prototypes]
        Prototype classes
    classes_ : array-like, shape = [n_classes]
        Array containing labels.
    See also
    --------
    GrlvqModel, GmlvqModel, LgmlvqModel
    """

    def __init__(self, prototypes_per_class=1, initial_prototypes=None,
                 max_iter=2500, gtol=1e-5, beta=2, C=None,
                 display=False, random_state=None):
        super(GlvqModel, self).__init__(prototypes_per_class=prototypes_per_class,
                                        initial_prototypes=initial_prototypes,
                                        max_iter=max_iter, gtol=gtol, display=display,
                                        random_state=random_state)
        self.beta = beta
        self.c = C
        #self.num = 0
        #self.iteracc = {}

    def phi(self, x):
        """
        Parameters
        ----------
        x : input value
        """
        return 1 / (1 + np.math.exp(-self.beta * x))

    def phi_prime(self, x):
        """
        Parameters
        ----------
        x : input value
        """
        return self.beta * np.math.exp(self.beta * x) / (
                1 + np.math.exp(self.beta * x)) ** 2

    def _iteracc(self, training_data, label_equals_prototype, prototypes):
        y_real = np.where(label_equals_prototype)[1]
        y_real = np.floor(y_real[::self.prototypes_per_class] / self.prototypes_per_class).astype(dtype=int)
        y_pred = self._compute_distance(training_data, prototypes)
        y_pred = self.c_w_[y_pred.argmin(1)].astype(dtype=int)
        acc = np.count_nonzero(y_real == y_pred) / len(y_real)
        print("Iteration Number: {:3} \n {} \n Accuracy: {} \n".format(
            self.num, prototypes, acc))
        #self.iteracc[self.num] = acc
        self.num += 1

    @staticmethod
    @njit
    def _optgradhelper(g, nb_prototypes, pidxcorrect, pidxwrong, mu, distwrong, distcorrect, distcorrectpluswrong,
                       training_data, prototypes):
        for i in range(nb_prototypes):
            idxc = i == pidxcorrect
            idxw = i == pidxwrong

            idxc_empty = np.all(idxc == False)
            idxw_empty = np.all(idxw == False)

            if not idxc_empty and not idxw_empty:
                dcd = mu[idxw] * distcorrect[idxw] * distcorrectpluswrong[idxw]
                dwd = mu[idxc] * distwrong[idxc] * distcorrectpluswrong[idxc]
                g[i] = dcd.dot(training_data[idxw]) - dwd.dot(training_data[idxc]) \
                       + (dwd.sum(0) - dcd.sum(0)) * prototypes[i]
            elif idxc_empty and idxw_empty:
                g[i] = np.zeros(training_data.shape[1])
            elif idxc_empty and not idxw_empty:
                dcd = mu[idxw] * distcorrect[idxw] * distcorrectpluswrong[idxw]
                g[i] = dcd.dot(training_data[idxw]) + (-dcd.sum(0)) * prototypes[i]
            else:
                dwd = mu[idxc] * distwrong[idxc] * distcorrectpluswrong[idxc]
                g[i] = -dwd.dot(training_data[idxc]) + dwd.sum(0) * prototypes[i]
        return g

    def _optgrad(self, variables, training_data, label_equals_prototype,
                 random_state):
        n_data, n_dim = training_data.shape
        nb_prototypes = self.c_w_.size
        prototypes = variables.reshape(nb_prototypes, n_dim)

        dist = _squared_euclidean(training_data, prototypes)
        d_wrong = dist.copy()
        d_wrong[label_equals_prototype] = np.inf
        distwrong = d_wrong.min(1)
        pidxwrong = d_wrong.argmin(1)

        d_correct = dist
        d_correct[np.invert(label_equals_prototype)] = np.inf
        distcorrect = d_correct.min(1)
        pidxcorrect = d_correct.argmin(1)

        distcorrectpluswrong = distcorrect + distwrong
        distcorectminuswrong = distcorrect - distwrong
        mu = distcorectminuswrong / distcorrectpluswrong
        mu = self.beta * np.exp(self.beta * mu) / (1 + np.exp(self.beta * mu)) ** 2
        #mu = np.vectorize(self.phi_prime)(mu)

        g = np.zeros(prototypes.shape)
        distcorrectpluswrong = 4 / distcorrectpluswrong ** 2

        g = self._optgradhelper(g, nb_prototypes, pidxcorrect, pidxwrong, mu, distwrong, distcorrect,
                                distcorrectpluswrong, training_data, prototypes)
        g[:nb_prototypes] = 1 / n_data * g[:nb_prototypes]
        g = g * (1 + 0.0001 * random_state.rand(*g.shape) - 0.5)

        #g = np.zeros(prototypes.shape)
        #for i in range(nb_prototypes):
        #    idxc = i == pidxcorrect
        #    idxw = i == pidxwrong
        #
        #    dcd = mu[idxw] * distcorrect[idxw] * distcorrectpluswrong[idxw]
        #    dwd = mu[idxc] * distwrong[idxc] * distcorrectpluswrong[idxc]
        #    g[i] = dcd.dot(training_data[idxw]) - dwd.dot(training_data[idxc]) \
        #           + (dwd.sum(0) - dcd.sum(0)) * prototypes[i]
        #
        #g[:nb_prototypes] = 1 / n_data * g[:nb_prototypes]
        #g = g * (1 + 0.0001 * random_state.rand(*g.shape) - 0.5)

        # display information
        #self._iteracc(training_data, label_equals_prototype, prototypes)
        return g.ravel()

    def _optfun(self, variables, training_data, label_equals_prototype):
        n_data, n_dim = training_data.shape
        nb_prototypes = self.c_w_.size
        prototypes = variables.reshape(nb_prototypes, n_dim)

        dist = _squared_euclidean(training_data, prototypes)
        d_wrong = dist.copy()
        d_wrong[label_equals_prototype] = np.inf
        distwrong = d_wrong.min(1)

        d_correct = dist
        d_correct[np.invert(label_equals_prototype)] = np.inf
        distcorrect = d_correct.min(1)

        distcorrectpluswrong = distcorrect + distwrong
        distcorectminuswrong = distcorrect - distwrong
        mu = distcorectminuswrong / distcorrectpluswrong
        [self._map_to_int(x) for x in self.c_w_[label_equals_prototype.argmax(1)]]
        mu *= self.c_[label_equals_prototype.argmax(1), d_wrong.argmin(1)]  # y_real, y_pred

        mu = 1 / (1 + np.exp(-self.beta * mu))
        return mu.sum(0)
        # return np.vectorize(self.phi)(mu).sum(0)

    def _validate_train_parms(self, train_set, train_lab):
        if not isinstance(self.beta, int):
            raise ValueError("beta must a an integer")

        ret = super(GlvqModel, self)._validate_train_parms(train_set, train_lab)

        self.c_ = np.ones((self.c_w_.size, self.c_w_.size))
        if self.c is not None:
            self.c = validation.check_array(self.c)
            if self.c.shape != (2, 3):
                raise ValueError("C must be shape (2,3)")
            for k1, k2, v in self.c:
                self.c_[tuple(zip(*product(self._map_to_int(k1), self._map_to_int(k2))))] = float(v)
        return ret

    def _map_to_int(self, item):
        return np.where(self.c_w_ == item)[0]

    def _optimize(self, x, y, random_state):
        label_equals_prototype = y[np.newaxis].T == self.c_w_
        res = minimize(
            fun=lambda vs: self._optfun(
                variables=vs, training_data=x,
                label_equals_prototype=label_equals_prototype),
            jac=lambda vs: self._optgrad(
                variables=vs, training_data=x,
                label_equals_prototype=label_equals_prototype,
                random_state=random_state),
            method='l-bfgs-b', x0=self.w_,
            options={'disp': self.display, 'gtol': self.gtol,
                     'maxiter': self.max_iter})
        self.w_ = res.x.reshape(self.w_.shape)
        self.n_iter_ = res.nit

    def _compute_distance(self, x, w=None):
        if w is None:
            w = self.w_
        return cdist(x, w, 'euclidean')

    def predict(self, x):
        """Predict class membership index for each input sample.
        This function does classification on an array of
        test vectors X.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        check_is_fitted(self, ['w_', 'c_w_'])
        x = validation.check_array(x)
        if x.shape[1] != self.w_.shape[1]:
            raise ValueError("X has wrong number of features\n"
                             "found=%d\n"
                             "expected=%d" % (self.w_.shape[1], x.shape[1]))
        dist = self._compute_distance(x)
        return (self.c_w_[dist.argmin(1)])

    def decision_function(self, x):
        """Predict confidence scores for samples.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : array-like, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
        """
        check_is_fitted(self, ['w_', 'c_w_'])
        x = validation.check_array(x)
        if x.shape[1] != self.w_.shape[1]:
            raise ValueError("X has wrong number of features\n"
                             "found=%d\n"
                             "expected=%d" % (self.w_.shape[1], x.shape[1]))
        dist = self._compute_distance(x)

        foo = lambda c: dist[:,self.c_w_ != self.classes_[c]].min(1) - dist[:,self.c_w_ == self.classes_[c]].min(1)
        res = np.vectorize(foo, signature='()->(n)')(self.classes_).T

        if self.classes_.size <= 2:
            return res[:,1]
        else:
            return res