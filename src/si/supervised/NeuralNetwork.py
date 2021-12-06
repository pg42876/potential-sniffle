from abc import ABC, abstractmethod
import numpy as np
from numpy.core.numeric import outer
from si.supervised.Model import Model
from si.util.Metrics import mse

class Layer(ABC):

    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input):
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, output_error, learning_rate):
        raise NotImplementedError

class Dense(Layer):

    def __init__(self, input_size, output_size):
        
        """
        Fully connecter layer
        """
        
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(1, output_size) #ou np.zeros((1, output_size))
    
    def setWeights(self, weights, bias):
        if (weights.shape != self.weights.shape):
            raise ValueError(f"Shapes mismatch {weights.shape} and")
        if (bias.shape != self.bias.shape):
            raise ValueError(f"Shapes mismatch {bias.shape} and")
        self.weights = weights
        self.bias = bias
    
    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_Error, learning_data):
        raise NotImplementedError
    
class Activation(Layer):
    
    def __init__(self, activation):

        """
        Activation layer.
        """

        self.ativation = activation
    
    def forward(self, input_data):
        self.input = input_data
        self.output = self.ativation(self.input)
        return self.output

    def backward(self, output_Error, learning_data):
        raise NotImplementedError

class NN(Model):

    def __init__(self, epochs = 1000, lr = 0.01, verbose = True):

        """
        Neural Network model
        """

        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose

        self.layers = []
        self.loss = mse

    def add(self, layer):
        self.layers.append(layer)
    
    def fit(self, dataset):
        raise NotImplementedError

    def predict(self, input_data):
        assert self.is_fitted, 'Model must be fit befora predicting'
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def cost(self, X = None, Y = None):
        assert self.is_fitted, 'Model must be fit befora predicting'
        X = X if X is not None else self.dataset.X
        Y = Y if Y is not None else self.dataset.Y
        output = self.predcit(X)
        return self.loss(X, output)