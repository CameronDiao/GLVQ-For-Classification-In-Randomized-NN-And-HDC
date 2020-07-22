# -*- coding: utf-8 -*-

# Author: Joris Jensen <jjensen@techfak.uni-bielefeld.de>
#
# License: BSD 3 clause

from __future__ import division

import time
import numpy as np
from numba import njit
from scipy.optimize import minimize

from sklearn.utils import validation
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import rbf_kernel
from itertools import product

from lvq import _LvqBaseModel

@njit
def _dist(k_matrix, x, w):
    """
    Calculates the feature space distance matrix between the (projected) data samples x
    and the prototype vectors w
    :param k_matrix: a kernel matrix of dimension M x M in which each element of index i, j
    represents the Mercer kernel function evaluated using data samples i, j of x
    In this instance, a numpy 2-D array
    :param x: a data matrix of dimension M x K containing data samples with their
    respective feature values
    In this instance, a numpy 2-D array
    :param w: a data matrix of dimension P x M containing prototype vectors represented by their
    respective 1 x M combinatorial coefficient vectors
    In this instance, a numpy 2-D array
    :return: dist_matrix: the feature space distance matrix between the (projected) data samples x
    and the prototype vectors w
    """
    dist_matrix = np.zeros((w.shape[0], x.shape[0]))
    k_diag = np.diag(k_matrix)
    for i in range(w.shape[0]):
        dist_matrix[i, :] = k_diag - 2 * np.sum(w[i, :] * k_matrix, axis=1) + \
            np.repeat(np.sum(np.outer(w[i, :], w[i, :]) * k_matrix), x.shape[0])
    return dist_matrix.T



class KglvqModel(_LvqBaseModel):
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
        super(KglvqModel, self).__init__(prototypes_per_class=prototypes_per_class,
                                        initial_prototypes=initial_prototypes,
                                        max_iter=max_iter, gtol=gtol, display=display,
                                        random_state=random_state)
        self.beta = beta
        self.c = C

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

    @staticmethod
    @njit
    def _optgradhelper(g, mu, n_data, pidxcorrect, pidxwrong, distcorrect, distwrong, distcorrectpluswrong,
                       prototypes):
        for i in range(n_data):
            idxc = pidxcorrect[i]
            idxw = pidxwrong[i]

            dcd = mu[idxw] * distcorrect[idxw] * distcorrectpluswrong[idxw]
            dwd = mu[idxc] * distwrong[idxc] * distcorrectpluswrong[idxc]

            g[idxw] -= dcd * prototypes[idxw]
            g[idxc] += dwd * prototypes[idxc]
            g[idxw, i] += dcd
            g[idxc, i] -= dwd
        return g

    def _optgrad(self, variables, training_data, label_equals_prototype,
                 random_state):
        """
        Computes the gradient vector of KGLVQ's objective cost function
        :param variables: a data matrix of dimension P x M containing prototype vectors represented by their
        respective 1 x M combinatorial coefficient vectors
        In this instance, a numpy 2-D array
        :param training_data: a data matrix of dimension M x K containing data samples with their
        respective feature values
        In this instance, a numpy 2-D array
        :param label_equals_prototype: a Boolean matrix of dimension M x P with truth values at each index i, j
        representing whether the class label of data sample i matches the class label of prototype vector j
        In this instance, a numpy 2-D array
        :param random_state: RandomState instance variable
        :return: g: a flattened numpy 1-D array representing the combined coefficient vectors of
        each prototype vector j
        """
        n_data, n_dim = training_data.shape
        nb_prototypes = self.c_w_.size
        prototypes = variables.reshape(nb_prototypes, n_data)
        k_matrix = rbf_kernel(training_data)

        dist = _dist(k_matrix, training_data, prototypes)
        d_wrong = dist.copy()
        d_wrong[label_equals_prototype] = np.inf
        distwrong = d_wrong.min(1)
        pidxwrong = d_wrong.argmin(1)

        d_correct = dist
        d_correct[np.invert(label_equals_prototype)] = np.inf
        distcorrect = d_correct.min(1)
        pidxcorrect = d_correct.argmin(1)

        del k_matrix
        del dist

        distcorrectpluswrong = distcorrect + distwrong
        distcorectminuswrong = distcorrect - distwrong
        mu = distcorectminuswrong / distcorrectpluswrong
        mu = self.beta * np.exp(self.beta * mu) / (1 + np.exp(self.beta * mu)) ** 2
        #mu = np.vectorize(self.phi_prime)(mu)


        g = np.zeros(prototypes.shape)
        distcorrectpluswrong = 4 / distcorrectpluswrong ** 2

        g = self._optgradhelper(g, mu, n_data, pidxcorrect, pidxwrong, distcorrect, distwrong,
                                distcorrectpluswrong, prototypes)
        g[:nb_prototypes] = 1 / n_data * g[:nb_prototypes]
        g = g * (1 + 0.0001 * random_state.rand(*g.shape) - 0.5)
        return g.ravel()

    def _optfun(self, variables, training_data, label_equals_prototype):
        """
        KGLVQ's objective cost function
        :param variables: a data matrix of dimension P x M containing prototype vectors represented by their
        respective 1 x M combinatorial coefficient vectors
        In this instance, a numpy 2-D array
        :param training_data: a data matrix of dimension M x K containing data samples with their
        respective feature values
        In this instance, a numpy 2-D array
        :param label_equals_prototype: label_equals_prototype: a Boolean matrix of dimension M x P with truth values
        at each index i, j representing whether the class label of data sample i matches the class label of
        prototype vector j
        In this instance, a numpy 2-D array
        :return: The cost function evaluated with respect to the input data samples/prototype vectors
        """
        n_data, n_dim = training_data.shape
        nb_prototypes = self.c_w_.size
        prototypes = variables.reshape(nb_prototypes, n_data)
        k_matrix = rbf_kernel(training_data)

        dist = _dist(k_matrix, training_data, prototypes)
        d_wrong = dist.copy()
        d_wrong[label_equals_prototype] = np.inf
        distwrong = d_wrong.min(1)

        d_correct = dist
        d_correct[np.invert(label_equals_prototype)] = np.inf
        distcorrect = d_correct.min(1)

        del k_matrix
        del dist

        distcorrectpluswrong = distcorrect + distwrong
        distcorectminuswrong = distcorrect - distwrong
        mu = distcorectminuswrong / distcorrectpluswrong
        [self._map_to_int(x) for x in self.c_w_[label_equals_prototype.argmax(1)]]
        mu *= self.c_[label_equals_prototype.argmax(1), d_wrong.argmin(1)]  # y_real, y_pred

        mu = 1 / (1 + np.exp(-self.beta * mu))
        return mu.sum(0)
        #return np.vectorize(self.phi)(mu).sum(0)

    def _validate_train_parms(self, train_set, train_lab):
        if not isinstance(self.beta, int):
            raise ValueError("beta must a an integer")

        ret = super(KglvqModel, self)._validate_train_parms(train_set, train_lab)

        self.c_ = np.ones((self.c_w_.size, self.c_w_.size))
        if self.c is not None:
            self.c = validation.check_array(self.c)
            if self.c.shape != (2, 3):
                raise ValueError("C must be shape (2,3)")
            for k1, k2, v in self.c:
                self.c_[tuple(zip(*product(self._map_to_int(k1), self._map_to_int(k2))))] = float(v)

        self.training_data = train_set
        if self.training_data is not None:
            self.training_data = validation.check_array(self.training_data)
        return ret

    def _map_to_int(self, item):
        return np.where(self.c_w_ == item)[0]

    def _optimize(self, x, y, random_state):
        label_equals_prototype = y[np.newaxis].T == self.c_w_
        #constraint = ({"type": "eq", "fun": lambda x: 1.0 - np.sum(x)})
        bounds = [(0, 0) if x == 0.0 else (0, 1) for x in list(self.w_.ravel())]
        res = minimize(
            fun=lambda vs: self._optfun(
                variables=vs, training_data=x,
                label_equals_prototype=label_equals_prototype),
            jac=lambda vs: self._optgrad(
                variables=vs, training_data=x,
                label_equals_prototype=label_equals_prototype,
                random_state=random_state),
            method='l-bfgs-b', x0=self.w_,
            bounds=bounds,
            options={'disp': self.display, 'gtol': self.gtol,
                     'maxiter': self.max_iter})
        self.w_ = res.x.reshape(self.w_.shape)
        self.n_iter_ = res.nit

    @staticmethod
    @njit
    def _compute_distance(k1, k2, k3, x, w):
        dist_matrix = np.zeros((w.shape[0], x.shape[0]))
        k_diag = np.diag(k1)
        for i in range(w.shape[0]):
            dist_matrix[i, :] = k_diag - 2 * np.sum(w[i, :] * k2, axis=1) + \
                                np.repeat(np.sum(np.outer(w[i, :], w[i, :]) * k3), x.shape[0])
        return dist_matrix.T

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
        self.training_data = validation.check_array(self.training_data)
        if x.shape[1] != self.training_data.shape[1]:
            raise ValueError("X has wrong number of features\n"
                             "found=%d\n"
                             "expected=%d" % (x.shape[1], self.training_data.shape[1]))
        dist = self._compute_distance(rbf_kernel(x), rbf_kernel(x, self.training_data),
                                      rbf_kernel(self.training_data), x, self.w_)
        return (self.c_w_[dist.argmin(1)])
