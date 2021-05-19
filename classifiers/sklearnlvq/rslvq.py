# -*- coding: utf-8 -*-

# Author: Joris Jensen <jjensen@techfak.uni-bielefeld.de>
#
# License: BSD 3 clause

from __future__ import division

import math
import numpy as np
from numba import njit, vectorize
from scipy.optimize import minimize
from sklearn.utils import validation
from sklearn.utils.validation import check_is_fitted
from .lvq import _LvqBaseModel


@vectorize
def _costfhelper(x, w, sigma):
    d = (x - w).reshape((1, -1)).T
    d = d.T.dot(d)
    return (-d / (2 * sigma))[0][0]


@njit
def _phelper(j, e, w_, c_w_, sigma, y=None, prototypes=None):
    if prototypes is None:
        prototypes = w_
    if y is None:
        fs = np.array([_costfhelper(e, prototypes[w], sigma) for w in range(prototypes.shape[0])])
    else:
        fs = np.array([_costfhelper(e, prototypes[i], sigma) for i in
                       range(prototypes.shape[0]) if
                       c_w_[i] == y])
    fs_max = max(fs)
    s = np.sum(np.exp(fs - fs_max))
    o = np.math.exp(
        _costfhelper(e, prototypes[j], sigma) - fs_max) / s
    return o


class RslvqModel(_LvqBaseModel):
    """Robust Soft Learning Vector Quantization
    Parameters
    ----------
    prototypes_per_class : int or list of int, optional (default=1)
        Number of prototypes per class. Use list to specify different
        numbers per class.
    initial_prototypes : array-like, shape =  [n_prototypes, n_features + 1],
     optional
        Prototypes to start with. If not given initialization near the class
        means. Class label must be placed as last entry of each prototype.
    sigma : float, optional (default=0.5)
        Variance for the distribution.
    max_iter : int, optional (default=2500)
        The maximum number of iterations.
    gtol : float, optional (default=1e-5)
        Gradient norm must be less than gtol before successful termination
        of bfgs.
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
    MrslvqModel, LmrslvqModel
    """

    def __init__(self, prototypes_per_class=1, initial_prototypes=None,
                 sigma=0.5, max_iter=2500, gtol=1e-5,
                 display=False, random_state=None):
        super(RslvqModel, self).__init__(prototypes_per_class=prototypes_per_class,
                                         initial_prototypes=initial_prototypes,
                                         max_iter=max_iter, gtol=gtol, display=display,
                                         random_state=random_state)
        self.sigma = sigma

    @staticmethod
    @njit
    def _optgradhelper(g, n_data, training_data, label_equals_prototype, prototypes, sigma,
                       c_w_, w_):
        for i in range(n_data):
            xi = training_data[i]
            c_xi = label_equals_prototype[i]
            for j in range(prototypes.shape[0]):
                d = (xi - prototypes[j])
                if c_w_[j] == c_xi:
                    g[j] += (1 / sigma) * (_phelper(j, xi, w_, c_w_, sigma, prototypes=prototypes, y=c_xi) -
                                           _phelper(j, xi, w_, c_w_, sigma, prototypes=prototypes)) * d
                else:
                    g[j] -= (1 / sigma) * _phelper(j, xi, w_, c_w_, sigma, prototypes=prototypes) * d
        return g

    def _optgrad(self, variables, training_data, label_equals_prototype,
                 random_state):
        n_data, n_dim = training_data.shape
        nb_prototypes = self.c_w_.size
        prototypes = variables.reshape(nb_prototypes, n_dim)

        g = np.zeros(prototypes.shape)
        g = self._optgradhelper(g, n_data, training_data, label_equals_prototype, prototypes, self.sigma,
                                self.c_w_, self.w_)
        g /= n_data
        g *= -(1 + 0.0001 * random_state.rand(*g.shape) - 0.5)
        return g.ravel()

    @staticmethod
    @njit
    def _optfunhelper(n_data, training_data, label_equals_prototype, prototypes, c_w_, sigma):
        out = 0
        for i in range(n_data):
            xi = training_data[i]
            y = label_equals_prototype[i]
            fs = np.array([_costfhelper(xi, prototypes[w], sigma) for w in range(prototypes.shape[0])])
            fs_max = max(fs)
            s1 = np.sum(np.array([np.math.exp(fs[i] - fs_max) for i in range(len(fs))
                                  if c_w_[i] == y]))
            s2 = np.sum(np.exp(fs - fs_max))
            s1 += 0.0000001
            s2 += 0.0000001
            out += math.log(s1 / s2)
        return out

    def _optfun(self, variables, training_data, label_equals_prototype):
        n_data, n_dim = training_data.shape
        nb_prototypes = self.c_w_.size
        prototypes = variables.reshape(nb_prototypes, n_dim)

        out = self._optfunhelper(n_data, training_data, label_equals_prototype, prototypes, self.c_w_,
                                 self.sigma)
        return -out

    def _optimize(self, x, y, random_state):
        label_equals_prototype = y
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

    def _costf(self, x, w, **kwargs):
        d = (x - w)[np.newaxis].T
        d = d.T.dot(d)
        return -d / (2 * self.sigma)

    def _p(self, j, e, y=None, prototypes=None, **kwargs):
        if prototypes is None:
            prototypes = self.w_
        if y is None:
            fs = [self._costf(e, w, **kwargs) for w in prototypes]
        else:
            fs = [self._costf(e, prototypes[i], **kwargs) for i in
                  range(prototypes.shape[0]) if
                  self.c_w_[i] == y]
        fs_max = max(fs)
        s = np.sum(np.exp(fs - fs_max))
        o = np.math.exp(
            self._costf(e, prototypes[j], **kwargs) - fs_max) / s
        return o

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

        def foo(e):
            fun = np.vectorize(lambda w: self._costf(e, w),
                               signature='(n)->()')
            pred = fun(self.w_).argmax()
            return self.c_w_[pred]

        return np.vectorize(foo, signature='(n)->()')(x)

    def posterior(self, y, x):
        """
        calculate the posterior for x:
         p(y|x)
        Parameters
        ----------

        y: class
            label
        x: array-like, shape = [n_features]
            sample
        Returns
        -------
        posterior
        :return: posterior
        """
        check_is_fitted(self, ['w_', 'c_w_'])
        x = validation.column_or_1d(x)
        if y not in self.classes_:
            raise ValueError('y must be one of the labels\n'
                             'y=%s\n'
                             'labels=%s' % (y, self.classes_))
        s1 = sum([self._costf(x, self.w_[i]) for i in
                  range(self.w_.shape[0]) if
                  self.c_w_[i] == y])
        s2 = sum([self._costf(x, w) for w in self.w_])
        return s1 / s2
