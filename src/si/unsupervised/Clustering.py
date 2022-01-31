import numpy as np
from scipy import stats
from  copy import copy
import warnings
from src.si.util.util import euclidean, manhattan

class Kmeans:
    """
    A cluster refers to a collection of data points aggregated together because of certain similarities.
    You’ll define a target number k, which refers to the number of centroids you need in the dataset.
    A centroid is the imaginary or real location representing the center of the cluster.

    The K-means algorithm identifies k number of centroids, and then allocates every data
    point to the nearest cluster, while keeping the centroids as small as possible.
    The ‘means’ in the K-means refers to averaging of the data; that is, finding the centroid.
    """
    def __init__(self, K: int, max_interactions = 100, distance = 'euclidean'):
        self.k = K #n elementos que queremos nos centroids
        self.max_interaction = max_interactions #numero maximo de iteracoes
        self.centroids = None #numero de centroides
        if distance == 'euclidean': #ver o tipo de distancia
            self.distance = euclidean
        elif distance == 'manhattan':
            self.distance = manhattan
        else: # caso nao seja nem euclidean nem manhattan
            raise Exception('Distance metric not available \n Score functions: euclidean, manhattan')
    
    def fit(self, dataset):
        """Randomly selects K centroids"""

        x = dataset.X
        self._min = np.min(x, axis = 0)#min de cada linha (guarda objeto): int
        self._max = np.max(x, axis = 0)#max de cada linha (guarda objeto): int

    def init_centroids(self, dataset):
        x = dataset.X
        cent =[]
        for i in range(x.shape[1]):#corre colunas
            cent.append(np.random.uniform(low=self._min[i], high=self._max[i], size=(self.k,)))
            #low: Lower boundary of the output interval. All values generated will be greater than or equal to low.
            #high: Upper boundary of the output interval. All values generated will be less than or equal to high.
            #size: Output shape. size=(self.k,): self.k * tudo
        self.centroids = np.array(cent).T # guarda num objeto os centroides em que e um array
        # T = transposta

    def get_closest_centroid(self, x):
        dist = self.distance(x, self.centroids)#distancia
        closest_centroid_index = np.argmin(dist, axis=0)
        #retorna os indices dos valores mais pequenos por linha
        # array([[10, 11, 12],[13, 14, 15]]) -> ex: axis = 0: array([0, 0, 0]), axis = 1, array([0, 0])
        return closest_centroid_index
    
    def transform(self, dataset):
        self.init_centroids(dataset)
        X = dataset.X.copy()
        changed = False
        count = 0
        old_idxs = np.zeros(X.shape[0])
        #data.shape() -> The elements of the shape tuple give the lengths of the corresponding array dimensions.
        while not changed or count < self.max_interaction: #while not changed == True
            idxs = np.apply_along_axis(self.get_closest_centroid, axis=0, arr=X.T)
            #vai aplicar get closeste centroid ao axis 0
            #axis: Axis along which arr is sliced.
            #arr: Input array.
            cent = []
            for i in range(self.k):
                cent.append(np.mean(X[idxs == i], axis=0))#calcular a media sobre os pontos e essas medias vao ser os novos pontos
            #cent = [np.mean(X[idxs == i],axis = 0) for i in range(self.k)]
            self.centroids = np.array(cent)
            changed = np.all(old_idxs == idxs) #Test whether all array elements along a given axis evaluate to True.
            old_idxs = idxs# in indexes antigos passam a ser os novos
            count += 1#aumenta a conta para o maximo de iteracoes
        return self.centroids, idxs

    def fit_transform(self, dataset):
        self.fit(dataset)
        centroides, idxs = self.transform(dataset)
        return centroides, idxs
