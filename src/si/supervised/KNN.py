from src.si.supervised.Model import Model
from src.si.util import euclidean
from src.si.util.metrics import accuracy_score
import numpy as np

class KNN(Model):
    def __init__(self, num_neighbors, classification=True):
        super(KNN).__init__()#invocar o init do modelo
        self.num_neighbors = num_neighbors
        self.classification = classification

    def fit(self, dataset):
        """"""
        self.dataset = dataset
        self.is_fitted = True

    def get_neighbors(self, x):# x e um numero
        distances = euclidean(x, self.dataset.X)
        sorted_index = np.argsort(distances)#os indices vao ser postos por ordem crescente
        return sorted_index[:self.num_neighbors]#ate aos neighboors especificados

    def predict(self, x):
        assert self.is_fitted, 'Model must be fit before predicting'
        neighbors = self.get_neighbors(x)#obtem os neighboors (pontos mais proximos)
        values = self.dataset.y[neighbors].tolist()#vai escolher os neighboors e passa para lista
        if self.classification:
            prediction = max(set(values), key=values.count)#retorna o que tem o valor maximo da label (a label que se repete mais vezes)
        else:
            prediction = sum(values) / len(values)
        return prediction

    def cost(self, X=None, y=None):
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y

        y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=X.T)
        # ma: temos de usar porque pode formatar a previsao
        return accuracy_score(y, y_pred)

