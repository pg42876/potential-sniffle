import numpy as np
from scipy import stats
from copy import copy
import warnings
from si import data
from si.data import Dataset
from si.util.Util import euclidean, manhattan, summary
from si.util.Scale import StandardScaler

class PCA:

    """
    Reduz a dimensão do dataset, mas mantém compacta a informação do dataset de maior dimensão
    Só podem ser usados dados numéricos
    """

    def __init__(self, components = 2, method = 'svd'):
        self.components = components 
        available_methods = ['svd', 'evd'] 
        if method not in available_methods:
            raise Exception(f"Method not available. Please choose between: {available_methods}.")
        self.method = method

    def tranform(self, dataset):
        xscale = StandardScaler().fit_transform(dataset) # normaliza os dados
        f = xscale.X.T
        if self.method == 'svd':
            self.vectors, self.values, rv = np.linalg.svd(f)
        else:
            matriz = np.cov(f)
            self.vectors, self.values, rv = np.linalg.svd(matriz)
        self.idxs_sort = np.argsort(self.values) # índices ordenados por importância dos componentes
        self.values_sort = self.values[self.idxs_sort] # ordena os valores pelo índice da coluna
        self.vectors_sort = self.vectors[:, self.idxs_sort] # ordena os vetores pelo índice da coluna (decrescente)
        if self.components > 0:
            if self.components > dataset.X.shape[1]:
                warnings.warn('The number of components is larger than the number of features.')
                self.components = dataset.X.shape[1]
            self.vector_comp = self.vectors_sort[:, 0:self.components] # vetores correspondentes ao número de componentes selecionados
        else:
            warnings.warn('The number of components is lower than 0.')
            self.components = 1
            self.vector_comp = self.vectors_sort[:, 0:self.components]
        r = np.dot(self.vector_comp.transpose(), f).transpose()
        return r

    def variance_transform(self):
        summary_value = np.sum(self.values_sort)
        evalues = []
        for value in self.values_sort:
            evalues.append(value / summary_value * 100)
        return np.array(evalues) # retorna um array com as variâncias em percentagem

class Kmeans:

    """
    Agrupa os dados tentando dividir as amostras por k grupos
    minimizando as distâncias entre pontos e centróides dos clusters.
    """

    def __init__ (self, K: int, max_interactions = 100, distance = 'euclidean'):
        self.k = K # número inteiro - número de clusters
        self.max_interactions = max_interactions # número máximo de interações
        self.centroids = None
        if distance == 'euclidean':
            self.distance = euclidean
        elif distance == 'manhattan':
            self.distance = manhattan
        else:
            raise Exception('Distance metric not available \n Score functions: euclidean, manhattan')

    def fit (self, dataset):

        """
        Adiciona ao self o mínimo e o máximo de todas os pontos
        """

        x = dataset.X
        self._min = np.min(x, axis = 0) # mínimo
        self._max = np.max(x, axis = 0) # máximo
        # não tem return porque estamos a guardar o resultado no objeto

    def init_centroids (self, dataset):

        """
        PRIMEIRA ITERAÇÃO
        Os primeiros centróides são encontrados de forma aleatória.
        """
        
        x = dataset.X
        centroids = []
        for c in range(x.shape[1]):
            centroids.append(np.random.uniform(low = self._min[c], high = self._max[c], size = (self.k))) # vai correr as colunas e avaliar todos os pontos (descobre os centróides)
        self.centroids = np.array(centroids).T # transforma em array e faz a transposta

    def get_closest_centroid (self, x):

        """
        Calcula as distâncias entre os centróides e escolhe as menores distâncias
        """

        dist = self.distance(x, self.centroids)
        closest_centroid_index = np.argmin(dist, axis = 0)
        return closest_centroid_index

    def transform (self, dataset):
        self.init_centroids(dataset) # primeiros centróides
        X = dataset.X 
        changed = False
        count = 0
        old_idxs = np.zeros(X.shape[0]) # array de zeros
        while not changed and count < self.max_interactions:
            idxs = np.apply_along_axis(self.get_closest_centroid, axis = 0, arr = X.T)
            centroids = []
            for i in range(self.k):
                centroids.append(np.mean(X[idxs == i], axis = 0)) # cálculo dos centróides com base na média (passam a ser os novos pontos)
            self.centroids = np.array(centroids) 
            changed = np.any(old_idxs == idxs) # o all vai testar se todos os valores; testa se todos os idxs são os antigos
            old_idxs = idxs # os idxs antigos passam a ser os novos
            count += 1 # aumenta o número de iterações
            return self.centroids, old_idxs

    def fit_transform (self, dataset):
        self.fit(dataset)
        centroids, idxs = self.transform(dataset)
        return centroids, idxs