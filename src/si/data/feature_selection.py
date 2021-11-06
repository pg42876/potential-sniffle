import numpy as np
from scipy import stats
from copy import copy
from .dataset import Dataset
import warnings

class VarianceThreshold:

    def __init__ (self, threshold = 0):

        #Serve para fazer a filtragem dos dados
        
        """  
        The variance threshold is a simple baseline approach to (...)
        It removes all features which variance doesn't meet some (...)
        """

        if threshold < 0:
            warnings.warn('The threshold must be a non-negative value.')
        self.threshold = threshold

    def fit (self, dataset):
        X = dataset.X
        self._var = np.var(X, axis = 0) #Calcula a variância dos valores dos datasets (variáveis não dependentes)

    def transform (self, dataset, inline = False):
        X = dataset.X
        cond = self._var > self.threshold #Guarda todas as variâncias -> array de bolianos
        idxs = []
        for a in range(len(cond)): #Seleção dos índices
            if cond[a]:
                idxs.append(a) #Se a variância for maior do que o threshold
        X_trans = X[:, idxs] #Seleção das features que nos interessam
        xnames = []
        for b in idxs:
            xnames.append(dataset._xnames[b]) #Seleção do nome das features (colunas)
        if inline: #Altera o dataset por completo. Se for True -> o dataset vai ser transformado com as novas condições
            dataset.X = X_trans
            dataset._xnames = xnames
            return dataset
        else: #Se for False -> criação de um novo dataset, existindo na mesma o velho
            return Dataset(X_trans, copy(dataset.Y), xnames, copy(dataset._yname))

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

class SelectKBest:
    
    def __init__(self, k, score_fun = 'f_regression'):
         if score_fun == 'f_regression':
