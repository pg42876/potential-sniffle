import numpy as np
import pandas as pd
from si.util.util import label_gen

__all__ = ['Dataset']

class Dataset:
    def __init__(self, X = None, Y = None,
                 xnames: list = None,
                 yname: str = None):

        """ Tabular Dataset """
        
        if X is None:
            raise Exception("Trying to instanciate a DataSet without any data")
        self.X = X #Linhas (dados independentes)
        self.Y = Y #Última linha do dataset de dados dependentes dos outros dados
        self._xnames = xnames if xnames else label_gen(X.shape[1])
        self._yname = yname if yname else 'Y'

    @classmethod
    def from_data(cls, filename, sep = ",", labeled = True):

        #Através de um ficheiro txt cria um dataset

        """Creates a DataSet from a data file.

        :param filename: The filename
        :type filename: str
        :param sep: attributes separator, defaults to ","
        :type sep: str, optional
        :return: A DataSet object
        :rtype: DataSet
        """

        data = np.genfromtxt(filename, delimiter = sep)
        if labeled: #Se a variável for igual a True; labeled -> para ver se tem a última coluna
            X = data[:, 0 : -1]
            Y = data[:, -1]
        else:
            X = data
            Y = None
        return cls(X, Y) #Vai transformar o (x, y) numa class dataset

    @classmethod
    def from_dataframe(cls, df, ylabel = None):

        """Creates a DataSet from a pandas dataframe.

        :param df: [description]
        :type df: [type]
        :param ylabel: [description], defaults to None
        :type ylabel: [type], optional
        :return: [description]
        :rtype: [type]
        """

        if ylabel is not None and ylabel in df.columns: #Verifica se existe variável dependente -> labeled = ylabel
            X = df.loc[:, df.columns != ylabel] #df.loc[:,] -> acessa às linhas e colunas pelo nome das colunas (df.columns != ylabel - menos a ylabel)
            Y = df.loc[:, ylabel].to_numpy() #Transforma para numpy array
            xnames = df.columns.tolist().remove(ylabel) #Remove o ylabel
            yname = ylabel
        else: #Se o ylabel não existir
            X = df.to_numpy() #Transforma em numpy array
            Y = None
            xnames = df.columns.tolist()
            yname = None
        return cls(X, Y, xnames, yname)

    def __len__(self):

        """ Returns the number of data points. """

        """
        Criar um função len para que sempre que chamarmos o len retornar o número de linhas para ele saber o que é suposto contar
        """

        return self.X.shape[0] #Devolve as linhas

    def hasLabel(self):

        """ Returns True if the dataset constains labels (a dependent variable) """

        return self.Y is not None #Número de labels; devolve True ou False (caso exista ou não variável dependente)

    def getNumFeatures(self):

        """ Returns the number of features """

        self.X.shape[1] #Retorna o número de colunas das variáveis independentes (1 representa as colunas)

    def getNumClasses(self):

        """ Returns the number of label classes or 0 if the dataset has no dependent variable. """

        return len(np.unique(self.Y)) if self.hasLabel() else 0 #Vai buscar os valores únicos das variáveis dependentes

    def writeDataset(self, filename, sep = ","):

        """ 
        Saves the dataset to a file
        :param filename: The output file path
        :type filename: str
        :param sep: The fields separator, defaults to ","
        :type sep: str, optional
        """

        fullds = np.hstack((self.X, self.Y.reshape(len(self.Y), 1))) #Junta x e y
        #RESHAPE -> y: lista com valores dependentes, mas vai tornar-se numa coluna com um elemento em cada linha
        np.savetxt(filename, fullds, delimiter = sep)

    def toDataframe(self):

        """ Converts the dataset into a pandas DataFrame """
        
        if self.Y is None: #Se não existir variável dependente (y)
            dataset = pd.DataFrame(self.X.copy(), coluns = self._xnames[:])
        else: #Se existir variável dependente
            dataset = pd.DataFrame(np.hstack((self.X, self.Y.reshape(len(self.Y), 1))), columns = np.hstack((self._xnames, self._yname)))
        return dataset

    def getXy(self): 

        """ Vai buscar os valores """

        return self.X, self.Y
