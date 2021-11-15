from model import Model
from si.util.util import manhattan, accuracy_score
import numpy as np

class KNN(Model):

    """
    
    """
    
    def __init__(self, number_neighboors, classification = True):
        super(KNN).__init__()
        self.number_neighboors = number_neighboors
        self.classification = classification

    def fit(self, dataset):
        self.dataset = dataset
        self.is_fitted = True

    def get_neighboors(self, x):

        """
        Calcula as distâncias entre cada ponto de teste
        em relação a todos os pontos do dataset de treino
        """

        distance = manhattan(x, self.dataset.X)
        idxs_sort = np.argsort(distance) #Dá sort aos idxs tendo em conta a distância
        return idxs_sort[:self.number_neighboors] #Retorna os idxs dos melhores pontos

    def predcit(self, x):

        """
        :param x: array de teste
        :return: predicted labels
        """

        assert self.is_fitted, 'Model must be fot before prediction'
        viz = self.get_neighboors(x) #Pontos mais próximos de x
        values = self.dataset.Y[viz].tolist() #Transforma os valores em lista
        if self.classification:
            prediction = max(set(values), key = values.count) #Avalia os valores máximos
        else:
            prediction = sum(values) / len(values)
        return prediction

    def cost(self):
        y_pred = np.ma.apply_along_axis(self.predcit, axis = 0, arr = self.dataset.X.T) #ma: máscara; temos de usar se não vai formatar a previsão
        return accuracy_score(self.dataset.Y, y_pred)

    