import numpy as np
from scipy import stats
from  copy import copy
import warnings
from scipy.stats import f_oneway, f
from .dataset import Dataset
__all__ = ['Dataset']

class VarianceThreshold:
    """
    The variance threshold is a simple baseline approach to feature selection.
    It removes all features which variance doesn't meet some threshold.
    """
    def __init__(self, threshold=0):
        """The variance threshold os a simple baseline approach to feat"""
        if threshold <0:
            warnings.warn("The thereshold must be a non-negative value.")
        self.threshold = threshold
    
    def fit(self, dataset):
        """Calcula a variancia"""
        X = dataset.X #variaveis nao dependentes
        self._var = np.var(X, axis=0)#aplica a riancia as variaveis n dependentes por linhas
        #self._var -> guarda na memoria do objeto um array com as variancias

    def transform(self, dataset, inline=False):
        """Escolhe as variancias que sao maiores que o threshold"""
        X = dataset.X
        cond = self._var > self.threshold #condicao retorna array de booleanos (True or False)
        idxs = [i for i in range(len(cond)) if cond[i]]
        #se a cond se verificar, ou seja, for True, vai fazer o append de i(o numero do index em que esta)

        X_trans = X[:,idxs] #:->todas as linhas, idxs -> features que me interessa manter que passaram a cond
        xnames = [dataset._xnames[i] for i in idxs]#buscar os nosmes das colunas em que a cond se verificou

        if inline:#se for True grava por cima do dataset existente
            dataset.X = X_trans #substituir as variaveis
            dataset._xnames = xnames #atualizo os nomes
            return dataset
        else:#se for False cria um Dataset novo
            return Dataset(X_trans, copy(dataset.Y), xnames, copy(dataset._yname))
    
    def fit_transform(self,dataset, inline=False):
        """Reduce X to the selected features."""
        self.fit(dataset)#corre o fit
        return self.transform(dataset, inline=inline)#corre o transform


class SelectKBest:
    """"
    Select features according to the K(Number of top features) highest scores
    (removes all but the K highest scoring features).
    """
    def __init__(self, K, score_function ="f_regression"):
        """
        Parameters
        ----------
        score_func: Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues)
        or a single array with scores.
        K: Number of top features to select."""

        available_score_function = ["f_classif", "f_regression"]
        
        #Ciclo para escolher funçao de score
        if score_function not in available_score_function:#confirmar se esta na lista de score_func possiveis
            raise Exception(f"Scoring function not available. Please choose between: {available_score_function}.")
        elif score_function == "f_classif":#ANOVA
            self.score_function = f_classif
        else:
            self.score_function = f_regression #Regressao de Pearson

        if K <= 0:#Valor de K tem que ser > 0
            raise Exception("The K value must be higher than 0.")
        else:
            self.k = K#guarda na memoria do objeto K

    def fit(self, dataset):
        """Run score function on dataset and get the appropriate features."""
        self.F_stat, self.p_value = self.score_function(dataset) #score_function = f_classif or f_regression
        #Retorna F_stat e o p-value

    def transform(self, dataset, inline=False):
        """Reduce X to the selected features."""
        X, X_names = copy(dataset.X), copy(dataset._xnames)

        if self.k > X.shape[1]:#se o K(numero de top features) for maior que o numero de features em X nao e possivel
            warnings.warn("The K value provided is greater than the number of features. "
                              "All features will be selected")
            self.k = int(X.shape[1])#passa a ser todas as features

        #Seleção de features
        select_features = np.argsort(self.F_stat)[-self.k:]
        #np.argsort(self.F_stat): retorna os indices que iriam por por ordem o array de acordo com o F score
        #[-self.k:]: vai buscar os indices; como é - vai buscar os ultimos porque queremos os com > score dependendo de K

        X_features = X[:, select_features] #:->todas as linhas, select_features -> features selecionadas
        #X_features_names = [X_names[index] for index in select_features]
        X_features_names = []
        for index in select_features:#vai buscar os nomes atraves dos indexes
            X_features_names.append(X_names[index])
        #X_features_names = [X_names[index] for index in select_features]

        if inline:#Se for True vai fazer a alteração do proprio dataset
            dataset.X = X_features
            dataset._xnames = X_features_names
            return dataset
        else:#Se for False faz um dataset novo
            return Dataset(X_features, copy(dataset.Y), X_features_names, copy(dataset._yname))

    def fit_transform(self, dataset, inline=False):
        """
        Fit to data, then transform it.
        Fits transformer to X and y and returns a transformed version of X.
        """
        self.fit(dataset)
        return self.transform(dataset, inline=inline)

def f_classif(dataset):
    """
    Scoring fucntion for classification. Compute the ANOVA F-value for the provided sample.
    :param dataset: A labeled dataset.
	:type dataset: Dataset.
	:return: F scores and p-value.
            statistic F: The computed F statistic of the test.
            _value: The associated p-value from the F distribution.
	        rtype_ a tupple of np.arrays.
	"""

    X, y = dataset.getXy()
    args = []
    for k in np.unique(y):#valores unicos em Y
        args.append(X[y == k, :])
    F_stat, p_value = stats.f_oneway(*args)#Perform one-way ANOVA.
        #The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean.
        #The test is applied to samples from two or more groups, possibly with differing sizes.
        #*args = sample1, sample2, …array_like:
            #The sample measurements for each group. There must be at least two arguments.
            #If the arrays are multidimensional, then all the dimensions of the array must be the same except for axis.
    return F_stat, p_value

def f_regression(dataset):
    """Scoring function for regressions.
        param dataset: A labeled dataset.
    	type dataset: Dataset.
    	return: F scores and p-value."""
    X, y = dataset.getXy()
    correlation_coef = np.array([stats.pearsonr(X[:,i], y)[0] for i in range(X.shape[1])])#X and y are array's
    degree_of_freedom = y.size - 2 #size: number of elements in the array (n de linha -2)
    corr_coef_squared = correlation_coef ** 2
    F_stat = corr_coef_squared / (1 - corr_coef_squared) * degree_of_freedom
    p_value = stats.f.sf(F_stat, 1, degree_of_freedom)
    #sf(x, dfn, dfd, loc=0, scale=1) -> Survival function (or reliability function or complementary cumulative distribution function):
                                        #The survival function is a function that gives the probability that a patient,
                                        #device, or other object of interest will survive beyond any specified time.
    #dnf -> Disjunctive normal form
    #dfd -> Degrees of freedom
    return F_stat, p_value
