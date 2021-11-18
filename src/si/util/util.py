import itertools
import numpy as np
import pandas as pd

# Y is reserved to idenfify dependent variables
ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXZ'

__all__ = ['label_gen', 'summary']

def label_gen(n):

    """ Generates a list of n distinct labels similar to Excel"""
    #Dá o nome às colunas

    def _iter_all_strings():
        size = 1
        while True:
            for s in itertools.product(ALPHA, repeat = size):
                yield "".join(s)
            size += 1
    generator = _iter_all_strings()

    def gen():
        for s in generator:
            return s
    return [gen() for _ in range(n)]


def summary(dataset, format = 'df'):

    """ Returns the statistics of a dataset(mean, std, max, min)

    :param dataset: A Dataset object
    :type dataset: si.data.Dataset
    :param format: Output format ('df':DataFrame, 'dict':dictionary ), defaults to 'df'
    :type format: str, optional
    """

    if dataset.hasLabel():
        data = np.hstack((dataset.X, dataset.Y.reshape(len(dataset.Y), 1))) #reshape -> serve para transformar em array; esta linha dá uma nova forma ao array (vai ter as linhas dataset.Y e uma coluna)
        names = []
        for d in dataset._xnames:
            names.append(d)
        names.append(dataset._yname) #Abre-se a lista names e juntam-se os nomes de xnames e de yname
    else:
        data = dataset.X.copy() #Para salvaguardar o dataset
        names = [dataset._xnames]
    mean = np.mean(data, axis = 0)
    var = np.var(data, axis = 0)
    maximo = np.max(data, axis = 0)
    minimo = np.min(data, axis = 0)
    stats = {}
    for i in range(data.shape[1]): #Guarda tudo num dicionário
        statistic = {'mean': mean[i], #Média da coluna
                    'var': var[i], #Variância da coluna
                    'max': maximo[i], #Máximo da coluna
                    'min': minimo[i] #Mínimo da coluna
                    }
        stats[names[i]] = statistic
    if format == 'df': #Transforma num dataframe
        df = pd.DataFrame(stats)
        return df
    else:
        return stats

def manhattan(x, y): #L1

    """

    """

    dist = (np.absolute(x - y)).sum(axis = 1)
    return dist

def euclidean(x, y): #L2

    """
    Distância entre dois pontos
    """

    dist = np.sqrt(np.sum((x - y) ** 2, axis = 1))
    return dist

def accuracy_score(y_true, y_pred):
    """
    Verifica quais são as previsões que são iguais a valores reais
    """
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    accuracy = correct / len(y_true)
    return accuracy

def train_test_split(dataset, split = 0.8):
    n = dataset.X.shape[0]
    m = int(split * n)
    array = np.arange(n)
    np.random.shuffle(array)
    from si.data.dataset import Dataset
    train = Dataset(dataset.X[array[:m]], dataset.Y[array[:m]], dataset._xnames, dataset._yname)
    test = Dataset(dataset.X[array[m:]], dataset.Y[array[m:]], dataset._xnames, dataset._yname)
    return train, test

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

