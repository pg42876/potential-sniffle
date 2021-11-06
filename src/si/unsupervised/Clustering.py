import numpy as np
from scipy import stats
from copy import copy
import warnings
from si.util.util import euclidean, l2_distance, manhattan

class PCA:
    def __init__(self):
        pass

    def fit(self, dataset):
        pass

    def tranform(self, dataset):
        pass

    def fit_transform(self, dataset):
        pass

    def variance_transform(self, dataset):
        pass


class Kmeans:

    def __init__ (self, K: int, max_interactions = 100, distance = 'euclidean'):
        self.k = K #Número inteiro
        self.max_interactions = max_interactions
        self.centroids = None
        if distance == 'euclidean':
            self.dist = euclidean
        elif distance == 'manhattan':
            self.distance = manhattan
        else:
            pass

    def fit (self, dataset):
        x = dataset.X
        self._min = np.min(x, axis = 0)
        self._max = np.max(x, axis = 0)
        #Não tem return porque estamos a guardar o resultado no objeto

    def init_centroids (self, dataset):
        x = dataset.X
        centroids = []
        for c in range(x.shape[1]):
            centroids.append(np.random.uniform(low = self._min[c], high = self._max[c], size = (self.k))) #Vai correr as colunas, self.k -> 
        self.centroids = np.array(centroids).T

    def get_closest_centroid (self, x):
        dist = self.distance(x, self.centroids)
        closest_centroid_index = np.argmin(dist, axis = 0)
        return closest_centroid_index

    def transform (self, dataset):
        self.init_centroids(dataset)
        print(self.centroids)
        X = dataset.X
        change = False
        count = 0
        old_idxs = np.zeros(X.shape[0]) #Array de zeros
        while not change or count < self.max_interactions:
            idxs = np.array_along_axis(self.get_closest_centroid, axis = 0, arr = X.T)
            centroids = []
            for i in range(self.k):
                centroids.append(np.mean(X[idxs == i], axis = 0)) #Cálculo dos centróides com base na média
            self.centroids = np.array(centroids) 
            change = np.all(old_idxs == idxs) #O all vai testar se todos os valores
            old_idxs = idxs
            count += 1 
            return self.centroids, old_idxs

    def fit_transform (self, dataset):
        self.fit(dataset)
        centroids, idxs = self.transform(dataset)
        return centroids, idxs