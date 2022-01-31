import numpy as np
from scipy import stats
from scipy.stats import f_oneway
from scipy.stats import f
from copy import copy
from .Dataset import Dataset
import warnings

class VarianceThreshold:

    def __init__ (self, threshold = 0):

        # serve para fazer a filtragem dos dados
        
        """  
        The variance threshold is a simple baseline approach to (...)
        It removes all features which variance doesn't meet some (...)
        """

        if threshold < 0:
            warnings.warn('The threshold must be a non-negative value.')
        self.threshold = threshold

    def fit (self, dataset):

        """
        Vai buscar todas as variáveis não dependentes e calcular a sua variância
        """
        
        X = dataset.X
        self._var = np.var(X, axis = 0) # self.var -> guarda os resultados na memória do objeto

    def transform (self, dataset, inline = False):
        X = dataset.X
        cond = self._var > self.threshold # guarda todas as variâncias -> array de bolianos (True ou False)
        idxs = []
        for a in range(len(cond)): # seleção dos índices
            if cond[a]: # is True:
                idxs.append(a) # faz o append do índice se a variância for maior do que o threshold
        X_trans = X[:, idxs] # seleção das features que nos interessam
        xnames = []
        for b in idxs:
            xnames.append(dataset.xnames[b]) # seleção do nome das features (colunas)
        if inline: # is True: ; altera o dataset por completo; se for True -> o dataset vai ser transformado com as novas condições (guarda por cima do dataset existente)
            dataset.X = X_trans
            dataset.xnames = xnames
            return dataset
        else: # se for False -> criação de um novo dataset, existindo na mesma o velho
            return Dataset(X_trans, copy(dataset.y), xnames, copy(dataset.yname)) # faz-se o copy com y porque estamos a ver as variáveis independentes

    def fit_transform(self, dataset, inline = False):
        self.fit(dataset) # recebe um dataset e corre o fit (variâncias) com esse dataset
        return self.transform(dataset, inline = inline)

class SelectKBest:

    """
    Semelhante à VarianceThreshold, mas em vez de trabalhar com variâncias trabalha com scores
    """

    def __init__(self, k, score_fun = 'f_regression'):
        if score_fun == 'f_regression':
            self.score_fun = f_regression
        elif score_fun == 'f_classification':
            self.score_fun == f_classification
        else:
            raise Exception("Score function not available \n Score functions: f_classification, f_regression")
        if k <= 0:
            raise Exception("K value invalid. K-value must be > 0") # número top selecionado (melhor valor)
        else:
            self.k = k

    def fit(self, dataset):
        self.Fscore, self.pvalue = self.score_fun(dataset) # vai buscar os valores da regressão de Pearson
    
    def transform(self, dataset, inline = False):
        data = copy(dataset.X)
        name = copy(dataset.xnames)
        if self.k > data.shape[1]: # self.k não pode ser maior que o número de colunas
            warnings.warn('K value greather than the number of features available.')
            self.k = data.shape[1] # tuplo com as dimensões do array
        lista = np.argsort(self.Fscore)[-self.k:] # sort das colunas pelo Fscore e depois vai buscar os índices; como temos (-) vai buscar os últimos valores porque queremos os que têm maior score, dependendo do k
        datax = data[:, lista] # dados das features selecionadas
        xnames = [name[ind] for ind in lista]
        if inline:
            dataset.X = datax
            dataset.xnames = xnames
            return dataset 
        else:
            return Dataset(datax, copy(dataset.y), xnames, copy(dataset.yname))
     
    def fit_transform(self, dataset, inline = False):
        self.fit(dataset) # recebe um dataset e corre o fit (scores) com esse dataset
        return self.transform(dataset, inline = inline)

def f_classification (dataset): 

    """
    ANOVA: avalia afirmações através das médias das populações.
    A análise permite verificar se exite ou não diferença significativa entre as
    médias e se os fatores têm influência nas variáveis dependentes.
    """

    X = dataset.X
    y = dataset.y
    aa = []
    for a in np.unique(y):
        aa.append(X[y == a, :]) 
    F_stat, pvalue = f_oneway(*aa)
    return F_stat, pvalue

def f_regression (dataset):

    """
    REGRESSÃO DE PEARSON: mede o grau da correlação entre duas variáveis de uma escala métrica.
    Varia entre -1 e 1:
    - p = 1: correlação perfeita positiva entre as duas variáveis;
    - p = -1: correlação perfeita negativa entre as duas variáveis, isto é, quando uma aumenta, a outra diminui;
    - p = 0: significa que as duas variáveis não dependem linearmente uma da outra.
    """

    X = dataset.X
    y = dataset.y
    cor_coef = np.array([stats.pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])
    dof = y.size - 2 # graus de liberdade
    cor_coef_sqrd = cor_coef ** 2
    F = cor_coef_sqrd / (1 - cor_coef_sqrd) * dof
    p = f.sf(F, 1, dof)
    return F, p