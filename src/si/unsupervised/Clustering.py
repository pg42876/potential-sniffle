import numpy as np
from scipy import stats
from copy import copy
import warnings
from si import data
from src.si.data import Dataset
from si.util.util import euclidean, manhattan, summary
from si.util.scale import StandardScaler

class PCA:

    """
    Reduz a dimensão do dataset, mas mantém compacta a informação do dataset de maior dimensão
    """

    def __init__(self, components = 2, method = 'svd'):
        self.components = components
        available_methods = ['svd', 'evd']
        if method not in available_methods:
            raise Exception(f"Method not available. Please choose between: {available_methods}.")
        self.method = method

    def tranform(self, dataset):
        xscale = StandardScaler().fit_transform(dataset) #Normaliza os dados
        f = xscale.X.T
        if self.method == 'svd':
            self.vectors, self.values, rv = np.linalg.svd(f)
        else:
            matriz = np.cov(f)
            self.vectors, self.values, rv = np.linalg.svd(matriz)
        self.idxs_sort = np.argsort(self.values) #Índices ordenados por importância dos componentes
        self.values_sort = self.values[self.idxs_sort] #Ordena os valores pelo índice da coluna
        self.vectors_sort = self.vectors[:, self.idxs_sort] #Ordena os vetores pelo índice da coluna
        if self.components > 0:
            if self.components > dataset.X.shape[1]:
                warnings.warn('The number of components is larger than the number of features.')
                self.components = dataset.X.shape[1]
            self.vector_comp = self.vectors_sort[:, 0:self.components] #Vetores correspondentes ao número de componentes selecionados
        else:
            warnings.warn('The number of components is lower than 0.')
            self.components = 1
            self.vector_comp = self.vectors_sort[:, 0:self.components]
        r = np.dot(self.vector_comp.transpose(), f).transpose()
        return r

    def fit_transform(self, dataset):
        s = self.tranform(dataset)
        summary_comp = self.variance_transform()
        return s, summary_comp

    def variance_transform(self):
        summary_value = np.sum(self.values_sort)
        evalues = []
        for value in self.values_sort:
            evalues.append(value / summary_value * 100)
        return np.array(evalues) #Retorna um array com as variâncias em percentagem

class Kmeans:

    """
    Agrupa os dados tentando dividir as amostras por k grupos
    minimizando as distâncias entre pontos e centróides dos clusters.
    """

    def __init__ (self, K: int, max_interactions = 100, distance = 'euclidean'):
        self.k = K #Número inteiro
        self.max_interactions = max_interactions #Número máximo de interações
        self.centroids = None
        if distance == 'euclidean':
            self.distance = euclidean
        elif distance == 'manhattan':
            self.distance = manhattan
        else:
            raise Exception('Distance metric not available \n Score functions: euclidean, manhattan')

    def fit (self, dataset):
        x = dataset.X
        self._min = np.min(x, axis = 0) #Mínimo
        self._max = np.max(x, axis = 0) #Máximo
        #Não tem return porque estamos a guardar o resultado no objeto

    def init_centroids (self, dataset):
        x = dataset.X
        centroids = []
        for c in range(x.shape[1]):
            centroids.append(np.random.uniform(low = self._min[c], high = self._max[c], size = (self.k))) #Vai correr as colunas e avaliar todos os pontos (descobre os centróides)
        self.centroids = np.array(centroids).T #Transforma em array e faz a transposta

    def get_closest_centroid (self, x):
        dist = self.distance(x, self.centroids)
        closest_centroid_index = np.argmin(dist, axis = 0)
        return closest_centroid_index

    def transform (self, dataset):
        self.init_centroids(dataset)
        X = dataset.X
        changed = False
        count = 0
        old_idxs = np.zeros(X.shape[0]) #Array de zeros
        while not changed or count < self.max_interactions:
            idxs = np.apply_along_axis(self.get_closest_centroid, axis = 0, arr = X.T)
            centroids = []
            for i in range(self.k):
                centroids.append(np.mean(X[idxs == i], axis = 0)) #Cálculo dos centróides com base na média (passam a ser os novos pontos)
            self.centroids = np.array(centroids) 
            changed = np.all(old_idxs == idxs) #O all vai testar se todos os valores; testa se todos os idxs são os antigos
            old_idxs = idxs #Os idxs antigos passam a ser os novos
            count += 1 #Aumenta o número de iterações
            return self.centroids, old_idxs

    def fit_transform (self, dataset):
        self.fit(dataset)
        centroids, idxs = self.transform(dataset)
        return centroids, idxs