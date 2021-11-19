from model import Model
from si.util.util import euclidean
from si.util.metrics import accuracy_score
import numpy as np

class KNN(Model):

    """
    K Vizinhos Mais Próximos (método lento na fase de previsão)
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

        distance = euclidean(x, self.dataset.X) #Calcular a distância entre um ponto e todos os outros
        idxs_sort = np.argsort(distance) #Dá sort aos idxs tendo em conta a distância
        return idxs_sort[:self.number_neighboors] #Retorna os idxs dos melhores pontos (vizinhos mais próximos)

    def predcit(self, x):

        """
        :param x: array de teste
        :return: predicted labels
        """

        assert self.is_fitted, 'Model must be fot before prediction'
        viz = self.get_neighboors(x) #Pontos mais próximos de x 
        values = self.dataset.Y[viz].tolist() #Transforma os valores em lista
        if self.classification: #Se for uma variável discreta/fatorial
            prediction = max(set(values), key = values.count) #Avalia os valores máximos -> CLASSIFICAÇÃO
        else: #Se for uma variável contínua/numérica
            prediction = sum(values) / len(values) #Média dos valores -> REGRRESSÃO
        return prediction 

    def cost(self):
        y_pred = np.ma.apply_along_axis(self.predcit, axis = 0, arr = self.dataset.X.T) #ma: máscara; temos de usar se não vai formatar a previsão
        return accuracy_score(self.dataset.Y, y_pred) #Dá a precisão 