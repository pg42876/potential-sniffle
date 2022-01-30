import numpy as np
import pandas as pd
from si.util.Util import label_gen

__all__ = ['Dataset']


class Dataset:
    def __init__(self, X=None, y=None,
                 xnames: list = None,
                 yname: str = None):
        """ Tabular Dataset"""
        if X is None:
            raise Exception("Trying to instanciate a DataSet without any data")
        self.X = X#linhas - dados independentes
        self.y = y#label dependente
        self.xnames = xnames if xnames else label_gen(X.shape[1])
        self.yname = yname if yname else 'y'

    @classmethod
    def from_data(cls, filename, sep=",", labeled=True):
        """Creates a DataSet from a data file.
        :param filename: The filename
        :type filename: str
        :param sep: attributes separator, defaults to ","
        :type sep: str, optional
        :return: A DataSet object
        :rtype: DataSet
        """
        data = np.genfromtxt(filename, delimiter=sep)#Numpy genfromtxt() to load the data from the text files, with missing values handled as specified.
        #fname: It is the file, filename, list, string, list of string, or generator to read.
        #If the filename is with the extension gz or bz2, then the file is decompressed.
        #Note: that generator should always return byte strings.
        #Strings in the list are treated as lines.
        #delimiter: optional. This is the string used to separate the values by default, any consecutive
        #whitespace that occurs acts as a delimiter.
        if labeled:#ve se tem uma label
            X = data[:, 0:-1]
            y = data[:, -1]
        else:
            X = data
            y = None
        return cls(X, y)#trasnforma X,Y numa classe Dataset

    @classmethod
    def from_dataframe(cls, df, ylabel=None):
        """Creates a DataSet from a pandas dataframe.
        :param df: [description]
        :type df: [type]
        :param ylabel: [description], defaults to None
        :type ylabel: [type], optional
        :return: [description]
        :rtype: [type]
        """

        if ylabel and ylabel in df.columns:#ver se tem a label dependente
            # df.loc -> acede as linhas e as colunas pelas label(s)
            X = df.loc[:, df.columns != ylabel].to_numpy()
            #df.columns != ylabel -> menos a ylabel
            #to_numpy() -> Convert the DataFrame to a NumPy array
            y = df.loc[:, ylabel].to_numpy()#ou df.loc[:,df.columns == ylabel].to_numpy()
            xnames = list(df.columns)#todos os nomes das colunas menos a ylabel
            xnames.remove(ylabel)#so ylabel
            yname = ylabel
        else:# caso nao tenha label
            X = df.to_numpy()#converte diretamente para numpy array
            y = None #nao tem label
            xnames = list(df.columns) #nomes das variaveis independentes
            yname = None
        return cls(X, y, xnames, yname)#trasnforma X,Y numa classe Dataset

    def __len__(self):
        """Returns the number of data points."""
        return self.X.shape[0]#retorna o numero de linhas

    def hasLabel(self):
        """Returns True if the dataset constains labels (a dependent variable)"""
        return self.y is not None#verifica se existe label (True or False)

    def getNumFeatures(self):
        """Returns the number of features"""
        return self.X.shape[1]#retorna o numero de colunas

    def getNumClasses(self):
        """Returns the number of label classes or 0 if the dataset has no dependent variable."""
        return len(np.unique(self.y)) if self.hasLabel() else 0 #retorna os valores existentes na label ou 0

    def writeDataset(self, filename, sep=","):
        """Saves the dataset to a file
        :param filename: The output file path
        :type filename: str
        :param sep: The fields separator, defaults to ","
        :type sep: str, optional
        """
        if self.y is not None:#confirma se o dataset tem label
            fullds = np.hstack((self.X, self.y.reshape(len(self.y), 1)))
            #np.hstack -> juntar X,y
            #reshape em self.Y.reshape(len(self.Y): linhas e 1))) coluna -> para dar uma nova forma ao array sem mudar a data
        else:#se o dataset nao tive label
            fullds = self.X
        np.savetxt(filename, fullds, delimiter=sep)

    def toDataFrame(self):
        """ Converts the dataset into a pandas DataFrame"""
        import pandas as pd
        if self.y is not None: # se não tiver label
            fullds = np.hstack((self.X, self.y.reshape(len(self.y), 1)))
            columns = self.xnames[:]+[self.yname]
        else: # caso tenha label
            fullds = self.X.copy() # self.X.copy() -> copia os dados das variáveis independentes
            columns = self.xnames[:] # columns=self._xnames[:] -> os nomes das colunas dessas variáveis
        return pd.DataFrame(fullds, columns=columns)

    def getXy(self):
        return self.X, self.y
