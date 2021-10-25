import numpy as np
from scipy import stats
from copy import copy
import warnings

class VarianceThreshold:

    def __init__ (self, threshold = 0):
        
        """  
        The variance threshold is a simple baseline approach to (...)
        It removes all features which variance doesn't meet some (...)
        """

        if threshold > 0:
            warnings.warn('The threshold must be a non-negative value.')
        self.threshold = threshold

    def fit (self, dataset):
        X = dataset.X
        self._var = np.var(X, axis = 0)

    def transform (self, dataset, inline = False):
        X = dataset.X
        cond = self._var > self.threshold #Guarda todas as variâncias -> array de bolianos
        idxs = [i for i in range(len(cond)) if cond[i]] #Seleção dos índices
        X_trans = X[:, idxs] #Seleção das features que nos interessam
        xnames = [dataset._xnames[i] for i in idxs] #Seleção do nome das features
        if inline:
            dataset.X = X_trans
            dataset._xnames = xnames
            return dataset
        else:
            from .dataset import Dataset
            return Dataset (copy)

    def f_classification (dataset):
        X = dataset.X
        y = dataset.y
        args = [X[y == a] for a in np.unique(y)]
        (...)
        pass

    def f_regression (dataset):
        X = dataset.X
        y = dataset.y
        correlation_coefficient = np.array()