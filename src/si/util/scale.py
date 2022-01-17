import numpy as np
from copy import copy


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.var = None
    """
    Standardize features by centering the mean to 0 and unit variance.
    The standard score of an instance is calculated by:
        z = (x - u) / s
    where u is the mean of the training data and s is the standard deviation.
    Standardizing data is often necessary before training many machine
    learning models to avoid problems like exploding/vanishing gradients and
    feature dominance.
    Attributes
    ----------
    _mean : numpy array of shape (n_features, )
        The mean of each feature in the training set.
    _var : numpy array of shape (n_features, )
        The variance of each feature in the training set.
    """

    def fit(self, dataset):
        """
        Calculate and store the mean and variance of each feature in the
        training set.
        Parameters
        ----------
        dataset : A Dataset OBJECT to be standardized
        """
        self.mean = np.mean(dataset.X, axis=0) #média em cada coluna
        self.var = np.var(dataset.X, axis=0) #var em cada coluna

    def transform(self, dataset, inline=False):
        """
        Standardize data by subtracting out the mean and dividing by
        standard deviation calculated during fitting.
        Parameters
        ----------
        :param inline:
        dataset : A Dataset object to be standardized
        Returns
        -------
        A Dataset object with standardized data.

        """
        Z = (dataset.X - self.mean)/np.sqrt(self.var)
        if inline:
            dataset.X = Z
            return dataset
        else:
            from src.si.data import Dataset
            return Dataset(Z, copy(dataset.Y), copy(dataset._xnames), copy(dataset._yname))

    def fit_transform(self, dataset, inline=False):
        """
        Calculate and store the mean and variance of each feature and
        standardize the data.
        Parameters
        ----------
        dataset : A Dataset object to be standardized
        Returns
        -------
        A Dataset object to with standardized data.
        """
        self.fit(dataset)
        return self.transform(dataset, inline=inline)

    def inverse_transform(self, dataset, inline=False):
        """
        Transform data back into original state by multiplying by standard
        deviation and adding the mean back in.
        Inverse standard scaler:
            x = z * s + u
        where s is the standard deviation, and u is the mean.
        Parameters
        ----------
        :param dataset : A standardized Dataset object
        :param inline:
        Returns
        -------
        Dataset object
        """
        volta = dataset.X * np.sqrt(self.var) + self.mean
        if inline:
            dataset.X = volta
            return dataset
        else:
            from src.si.data import Dataset
            return Dataset(volta, copy(dataset.Y), copy(dataset._xnames), copy(dataset._yname))
