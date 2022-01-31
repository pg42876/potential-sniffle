import numpy as np
import pandas as pd
from si.util.util import label_gen

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
        self._xnames = xnames if xnames else label_gen(X.shape[1])
        self._yname = yname if yname else 'y'

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

    def toDataframe(self):
        """ Converts the dataset into a pandas DataFrame"""
        import pandas as pd
        if self.y is not None:#se nao tiver label
            fullds = np.hstack((self.X, self.y.reshape(len(self.y), 1)))
            columns = self._xnames[:]+[self._yname]
        else:#caso tenha label
            fullds = self.X.copy()#self.X.copy() -> copia os dados das variaveis independentes
            columns = self._xnames[:]#columns=self._xnames[:] -> os nomes das colunas dessas variaveis
        return pd.DataFrame(fullds, columns=columns)

    def getXy(self):
        return self.X, self.y


def summary(dataset, format='df'):
    """ Returns the statistics of a dataset(mean, std, max, min)

    param dataset: A Dataset object
    type dataset: si.data.Dataset
    param format: Output format ('df':DataFrame, 'dict':dictionary ), defaults to 'df'
    type format: str, optional
    """
    if dataset.hasLabel():#verifica se existe label
        data = np.hstack((dataset.X, dataset.y.reshape(len(dataset.y),1)))
        #np.hstack(junta X,Y) -> reshape em self.Y.reshape(len(self.Y): linhas e 1))) coluna
        names = []#lista com nome das colunas
        for i in dataset._xnames:
            names.append(i)
        names.append(dataset._yname)#adicionar o nome da coluna label
    else:#se nao tiver label
        data = dataset.X.copy()
        names = [dataset._xnames]# -> names = [[dataset._xnames]]: nomes das colunas das variaveis indepedentes
    mean = np.mean(data, axis=0)#axis 0 = rows, axis 1 = columns
    var = np.var(data, axis=0)
    maxim = np.max(data, axis=0)
    minim = np.min(data, axis=0)
    stats = {}
    #-> stats ={names[i]:{'mean': mean[i],'var': var[i],'max': maxi[i], 'min': mini[i]}, names[i]:{'mean': mean[i],'var': var[i],'max': maxi[i], 'min': mini[i]}}
    for i in range(data.shape[1]):#percorre as colunas
        stat = {'mean': mean[i]#faz a media da coluna i
            ,'var': var[i]#faz a variancia da coluna i
            ,'max': maxim[i]#faz o maximo da coluna i
            ,'min': minim[i]}#faz o minimo da coluna i

        stats[names[i]] = stat #key: names[i], value: stat
    if format == 'df':#se quiser em pandas dataframe
        df = pd.DataFrame(stats)#convert an array to a dataframe
        return df
    else:#se nao quiser
        return stats #retorna o dicionario stats
