import numpy as np
from scipy import stats
from copy import copy
import warnings
from si.util.util import l2_distance

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
    def __init__ (self, K: int, max_interactions = 100) -> None:
        self.k = K
        self.max_interactions = max_interactions
        self.centroids = None
        self.distance = l2_distance

    def fit (self, dataset):
        x = dataset.X
        self._min = np.min(x, axis = 0)
        self._max = np.max(x, axis = 0)

    def init_centroids (self, dataset):
        x = dataset.X
        self.centroids = np.array(
            [np.random.uniform(
                low = self._min[i], high = self._max[i], size = (self.k)
            ) for i in range (x.shape[1])]).T

    def get_closest_centroid (self, x):
        dist = self.distance(x, self.centroids)
        closest_centroid_index = np.argmin(dist, axis = 0)
        return closest_centroid_index

    def transform (self, dataset):
        self.init_centroids(dataset)
        print(self.centroids)
        X = dataset.X
        changed = True
        count = 0
        old_idxs = np.zeros(X.shape[0])
        while changed or count < self.max_interactions:
            idxs = np.array_along_axis(self.get_closest_centroid, axis = 0, arr = X.T)
            cent = [np.mean(X[idxs == i], axis = 0) for i in range(self.k)] #Cálculo dos centróides com base na média
            self.centroids = np.array(cent) 
            changed = np.all(old_idxs == idxs) #Quando esta condição não se verificar, o changed fica False e o ciclo para 
            old_idxs = idxs
            count += 1 
            return self.centroids, idxs

    def fit_transform (self, dataset):
        self.fit(dataset)
        pass