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
    Distância entre dois pontos é dada pela soma das diferenças absolutas das suas coordenadas.
    """

    dist = (np.absolute(x - y)).sum(axis = 1)
    return dist

def euclidean(x, y): #L2

    """
    Distância entre dois pontos é dada 
    """

    dist = np.sqrt(np.sum((x - y) ** 2, axis = 1))
    return dist

def train_test_split(dataset, split = 0.8):
    n = dataset.X.shape[0]
    m = int(split * n)
    array = np.arange(n)
    np.random.shuffle(array)
    from src.si.data.Dataset import Dataset
    train = Dataset(dataset.X[array[:m]], dataset.Y[array[:m]], dataset._xnames, dataset._yname)
    test = Dataset(dataset.X[array[m:]], dataset.Y[array[m:]], dataset._xnames, dataset._yname)
    return train, test

def sigmoide(z):
    return 1 / (1 + np.exp(- z))

def add_intersect(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))

def to_categorical(y, num_classes = None, dtype = 'float32'):
    y = np.array(y, dtype = 'int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[: -1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype = dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def minibatch(X, batchsize = 256, shuffle = True):
    N = X.shape[0]
    ix = np.arange(N)
    n_batches = int(np.ceil(N / batchsize))
    if shuffle:
        np.random.shuffle(ix)

    def mb_generator():
        for i in range(n_batches):
            yield ix[i * batchsize: (i + 1) * batchsize]
    return mb_generator()

