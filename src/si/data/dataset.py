import numpy as np
import pandas as pd
from si.util.util import label_gen

__all__ = ['Dataset']

class Dataset:
    def __init__(self, X = None, Y = None,
                 xnames: list = None,
                 yname: str = None):

        """ Tabular Dataset"""
        
        if X is None:
            raise Exception("Trying to instanciate a DataSet without any data")
        self.X = X #Linhas
        self.Y = Y #Colunas
        self.xnames = xnames if xnames else label_gen(X.shape[1])
        self.yname = yname if yname else 'Y'

    @classmethod
    def from_data(cls, filename, sep = ",", labeled = True):

        """Creates a DataSet from a data file.

        :param filename: The filename
        :type filename: str
        :param sep: attributes separator, defaults to ","
        :type sep: str, optional
        :return: A DataSet object
        :rtype: DataSet
        """

        data = np.genfromtxt(filename, delimiter = sep)
        if labeled:
            X = data[:, 0 : -1]
            Y = data[:, -1]
        else:
            X = data
            Y = None
        return cls(X, Y)

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

        if ylabel is not None and ylabel in df.columns:
            X = df.loc[:, df.columns != ylabel]
            Y = df.loc[:, ylabel].to_numpy()
            xnames = df.columns.tolist().remove()
            yname = ylabel
        else:
            X = df.to_numpy()
            Y = None
            xnames = df.columns.tolist()
            yname = None
        return cls(X, Y, xnames, yname)

    def __len__(self):

        """ Returns the number of data points. """

        return self.X.shape[0]

    def hasLabel(self):

        """ Returns True if the dataset constains labels (a dependent variable) """

        return self.Y is not None #Número de labels 

    def getNumFeatures(self):

        """ Returns the number of features """

        self.X.shape[1]

    def getNumClasses(self):

        """ Returns the number of label classes or 0 if the dataset has no dependent variable. """

        return len(np.unique(self.Y)) if self.hasLabel() else 0

    def writeDataset(self, filename, sep = ","):

        """ Saves the dataset to a file
        :param filename: The output file path
        :type filename: str
        :param sep: The fields separator, defaults to ","
        :type sep: str, optional
        """

        fullds = np.hstack((self.X, self.Y.reshape(len(self.Y), 1))) #Número de linhas igua
        np.savetxt(filename, fullds, delimiter = sep)

    def toDataframe(self):

        """ Converts the dataset into a pandas DataFrame """
        
        if self.Y in None: #Se não existir variável independente
            df = pd.DataFrame(self.X.copy(), coluns = self._xnames[:])
        else:
            df = pd.DataFrame(np.hstack((self.X, self.Y.reshape(len(self.Y), 1))), columns = np.hstack((self._xnames, self._yname)))
        return df

    def getXy(self): 

        """ Vai buscar os valores """

        return self.X, self.Y
