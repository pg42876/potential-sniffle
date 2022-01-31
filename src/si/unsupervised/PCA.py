import numpy as np
from scipy import stats
from  copy import copy
import warnings
from si.data.scale import StandardScaler

class PCA:
    """
    Consta de um procedimento algébrico que converte as variáveis originais (tipicamente correlacionadas) num conjunto
        de variáveis não correlacionadas (linearmente) que se designam por componentes principais (PC) ou variáveis
        latentes. Análise baseada na covariância das diversas variáveis.
        As PCs são ordenadas pela quantidade decrescente de variabilidade (variância) que explicam
        """

    def __init__(self, n_components=2, using="svd"):
        self.n_components = n_components

        available = "svd"
        if using != available:
            raise Exception(f"Method not available. Please choose: {available}.")
        self.using = using

    def fit(self, dataset):
        """NAO E PRECISO"""
        pass

    def transform(self, dataset):
        """
        Parameters
        ----------
        dataset : A Dataset object
        Returns
        -------
        A Dataset object with features selected.
        """
        X_scaled = StandardScaler().fit_transform(dataset)#Faz a normalização por standart scaler
        features_Scaled = X_scaled.X.T #features passam para as linhas
        if self.using == "svd":
            self.vectors, self.values, rv = np.linalg.svd(features_Scaled)
        else:
            cov_matrix = np.cov(features_Scaled)#estima a covariancia -> array com a covariancia das matrizes
            self.values, self.vectors = np.linalg.eig(cov_matrix)#valores proprios, vetores proprios normalizados
        
        self.sorted_idxs = np.argsort(self.values)[::-1]#-1 : > para o <
        #Gera uma lista: com os indexes das colunas ordenadas por importancia de componte, > para o < : self.sorted_idxs: [0 1 2 3]

        self.s_value = self.values[self.sorted_idxs]
        #Gera uma lista: Colunas dos valores e vetores sao reordenadas pelos indexes das colunas

        self.s_vector = self.vectors[:, self.sorted_idxs]#todas as listas(:), com aqueles indixes
        #Gera uma lista de listas:

        if self.n_components not in range(0, dataset.X.shape[1]+1):
            warnings.warn('Number of components is not between 0 and '+str(dataset.X.shape[1]))
            self.n_components = dataset.X.shape[1]
            warnings.warn('Number of components defined as ' + str(dataset.X.shape[1]))
        self.vetor_subset = self.s_vector[:, 0:self.n_components]
        #gera um conjunto apartir dos vetores e values ordenados: todas as listas, mas com aqueles componentes

        X_reduced = np.dot(self.vetor_subset.transpose(), features_Scaled).transpose()
        return X_reduced

    def explained_variances(self):
        """
        Returns
        -------
        Array with the explained variances in %.
        """
        soma = np.sum(self.s_value)
        percent = []
        for i in self.s_value:
            conta = (i/soma) * 100
            percent.append(conta)
        return np.array(percent)
        #self.values_subset = self.s_value[0:self.n_components]
        #return np.sum(self.values_subset), self.values_subset

    def fit_transform(self,dataset):
        """
        Parameters
        ----------
        dataset : A Dataset object
        Returns
        -------
        A Dataset object with features selected and Array with the explained variances in %.
        """
        x_reduced = self.transform(dataset)
        e_var = self.explained_variances()#variancias
        return x_reduced, e_var
