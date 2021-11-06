import numpy as np
from scipy import stats
from scipy.stats import f_oneway
from scipy.stats import f
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

    def fit_transform(self, dataset, inline = False):
        self.fit(dataset)
        return self.transform(dataset, inline = inline)

class SelectKBest:

    def __init__(self, k, score_fun = 'f_regression'):
        if score_fun == 'f_regression':
            self.score_fun = f_regression
        elif score_fun == 'f_classification':
            self.score_fun == f_classification
        else:
            raise Exception("Score function not available \n Score functions: f_classification, f_regression")
        if k <= 0:
            raise Exception("K value invalid. K-value must be > 0") #Número top selecionado (melhor valor)
        else:
            self.k = k

    def fit(self, dataset):
        self.Fscore, self.pvalue = self.score_fun(dataset) #Vai buscar os valores da regressão de Pearson
    
    def transform(self, dataset, inline = False):
        data = copy(dataset.X)
        name = copy(dataset._xnames)
        if self.k > data.shape[1]:
            warnings.warn('K value greather than the number of features available.')
            self.k = data.shape[1]
        lista = np.argsort(self.Fscore)[-self.k:]
        datax = data[:, lista] #Dados das features selecionadas
        xnames = [name[ind] for ind in lista]
        if inline:
            dataset.X = datax
            dataset.xnames = xnames
            return dataset 
        else:
            return Dataset(datax, copy(dataset.Y), xnames, copy(dataset.yname))
     
    def fit_transform(self, dataset, inline = False):
        self.fit(dataset)
        return self.transform(dataset, inline = inline)

def f_classification (dataset): 

    """ ANOVA: """

    X = dataset.X
    y = dataset.y
    aa = []
    for a in np.unique(y):
        aa.append(X[y == a, :]) 
    F_stat, pvalue = f_oneway(*aa)
    return F_stat, pvalue

def f_regression (dataset):

    """ REGRESSÃO DE PEARSON: """

    X = dataset.X
    y = dataset.y
    cor_coef = np.array([stats.pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])
    dof = y.size - 2  #Graus de liberdade
    cor_coef_sqrd = cor_coef ** 2
    F = cor_coef_sqrd / (1 - cor_coef_sqrd) * dof
    p = f.sf(F, 1, dof)
    return F, p