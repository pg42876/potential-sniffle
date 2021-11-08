from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self):
        """
        Abstract class defining an interface
        for supervised learning models.
        """
        self.is_fitted = False

    @abstractmethod
    def fit(self, dataset):
        raise NotImplementedError

    @abstractmethod
    def predcit(self, x):
        raise NotImplementedError

    @abstractmethod
    def cost(self):
        raise NotImplementedError
