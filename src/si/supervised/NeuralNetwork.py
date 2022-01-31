import numpy as np
from abc import ABC, abstractmethod
from si.supervised.Model import Model
from si.util.Metrics import mse, mse_prime
from si.util.im2col import pad2D, im2col, col2im
from si.util.Activation import *

class Layer(ABC):

    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input): # refers to the calculation and storage of intermediate variables for a neural network in order from the input layer to the output layer
        raise NotImplementedError

    @abstractmethod
    def backward(self, erro, learning_rate): # from the output to the input layer, according to the chain rule from calculus.
        raise NotImplementedError

class Dense(Layer):
   
    """
    In any neural network, a dense layer is a layer that is deeply connected with its preceding layer
    which means the neurons of the layer are connected to every neuron of its preceding layer.
    """

    def __init__(self, input_size, output_size):
        super(Dense, self).__init__()
        self.weights = np.random.rand(input_size, output_size) - 0.5 # np.random.rand(2,2) -0.5 = array([[X,Y], [R,F]])-0.5 = array([[a,b], [c,d]])
        self.bias = np.zeros((1, output_size)) # array([[0,0]])

    def setWeights(self, weights, bias): # weights, bias = arrays
        if (weights.shape != self.weights.shape): # confirmar que tem a mesma forma weigths.shape = (2,2) entao self.weigths.shape = (2,2)
            raise ValueError(f"Shapes mismatch {weights.shape} and {self.weights.shape}")
        if (bias.shape != self.bias.shape): # confirmar que tem a mesma forma bias.shape = (1,2) entao self.bias.shape = (1,2)
            raise ValueError(f"Shapes mismatch {bias.shape} and {self.bias.shape}")
        self.weights = weights # array
        self.bias = bias # array

    def forward(self, input):
        self.input = input # array
        self.output = np.dot(self.input, self.weights) + self.bias # (prdout of input* weigths) + bias
        return self.output

    def backward(self, output_error, learning_rate):
        weights_error = np.dot(self.input.T, output_error)
        bias_error = np.sum(output_error, axis = 0)
        input_error = np.dot(output_error, self.weights.T)

        # Update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error
        return input_error


class Activation(Layer):
    
    """
    Activation function decides, whether a neuron should be activated or not by calculating weighted sum and further adding bias with it.
    The purpose of the activation function is to introduce non-linearity into the output of a neuron
    (allow such networks to compute nontrivial problems using only a small number of nodes).
    """
    
    def __init__(self, func):
        super(Activation, self).__init__()
        self.activation = func

    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        """In a neural network, we would update the weights and biases of the neurons on the basis of the error at the output.
        This process is known as back-propagation.
        Activation functions make the back-propagation possible since the gradients are supplied along with the error to update the weights and biases."""
        return np.multiply(self.activation.prime(self.input), output_error)


class NN(Model): # neural networks

    def __init__(self, epochs = 1000, lr = 0.01, verbose = True):
        super(NN, self).__init__()
        self.epochs = epochs # defines the number times that the learning algorithm will work through the entire training dataset.
        self.lr = lr # Learning Rate
        self.verbose = verbose # want to see the output of your Nural Network while it's training

        self.layers = [] # layers
        self.loss = mse
        self.loss_prime = mse_prime

    def useLoss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def add(self, layer): # acrescentar layer
        self.layers.append(layer)

    def fit(self, dataset):
        
        """
        Fitting is an automatic process that makes sure your machine learning models have the individual
        parameters best suited to solve your specific real-world business problem with a high level of accuracy.
        """
        
        self.dataset = dataset
        X, y = dataset.getXy()
        self.history = dict()
        for epoch in range(self.epochs): # predict por epoch
            output = X
            # forward propagation
            for layer in self.layers:
                output = layer.forward(output)

            # backward propagation
            error = self.loss_prime(y, output)
            for layer in reversed(self.layers):
                error = layer.backward(error, self.lr)

            # calculate average error on all samples
            err = self.loss(y, output) # mse(Y, output) Y = true label, output = predicted label
            self.history[epoch] = err # guardar o erro por epoch
            if self.verbose:
                print(f"epoch {epoch+1} / {self.epochs}, error = {err}")
            else:
                print(f"epoch {epoch + 1} / {self.epochs}, error = {err}", end ='\r')

    def fit_batch(self, dataset, batchsize = 256): # The batchsize is a hyperparameter that defines the number of samples to work through before updating the internal model parameters.
        X, y = dataset.getXy()
        if batchsize > X.shape[0]:
            raise Exception('Number of batchs superior to length of dataset')
        n_batches = int(np.ceil(X.shape[0] / batchsize))
        self.dataset = dataset
        self.history = dict()
        for epoch in range(self.epochs):
            self.history_batch = np.zeros((1, batchsize))
            for batch in range(n_batches):
                output = X[batch * batchsize:(batch + 1) * batchsize, ]
                # X[vai buscar o numero de linhas]
                # The batch size defines the number of samples that will be propagated through the network.
                
                # forward propagation
                for layer in self.layers:
                    output = layer.forward(output)

                # backward propagation
                error = self.loss_prime(y[batch * batchsize:(batch + 1) * batchsize, ], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, self.lr)

                # calcule average error
                err = self.loss(y[batch * batchsize:(batch + 1) * batchsize, ], output)
                self.history_batch[0, batch] = err
            self.history[epoch] = np.average(self.history_batch)
            if self.verbose:
                print(f'epoch {epoch + 1} / {self.epochs}, error = {self.history[epoch]}')
            else:
                print(f"epoch {epoch + 1} / {self.epochs}, error = {self.history[epoch]}", end ='\r')
        self.is_fitted = True

    def predict(self, x):
        assert self.is_fitted, "Model must be fitted before prediction"
        output = x # data para obter o output
        for layer in self.layers: # self.layers e uma lista
            output = layer.forward(output)
        return output

    def cost(self, X = None, Y = None):

        """
        Technique of evaluating “the performance of our algorithm/model”.
        It takes both predicted outputs by the model and actual outputs and calculates how much wrong the model was in its prediction.
        """
        
        assert self.is_fitted, "Model must be fitted before prediction"
        X = X if X is not None else self.dataset.X
        Y = Y if Y is not None else self.dataset.Y
        output = self.predict(X) # array
        return self.loss(Y, output) # mse(Y, output) Y = true label, output = predicted label


class Flatten(Layer):
    
    """
    Flattening is converting the data into a 1-dimensional array for inputting it to the next layer.
    We flatten the output of the convolutional layers to create a single long feature vector.
    """
    
    def forward(self, input):
        self.input_shape = input.shape
        # flattern all but the 1st dimension
        output = input.reshape(input.shape[0], -1)
        return output

    def backward(self, erro, learning_rate):
        return erro.reshape(self.input_shape)


class Conv2D(Layer):
    
    """
    Convolution is a mathematical operation on two objects to produce an outcome that expresses how the shape of one
    is modified by the other. With this computation, we detect a particular feature from the input image and get the
    result having information about that feature.
    """
    
    def __init__(self, input_shape, kernel_shape, layer_depth, stride = 1, padding = 0):
        self.input_shape = input_shape
        self.in_ch = input_shape[2]
        self.out_ch = layer_depth
        self.stride = stride # move the number of pixels we want (doesn't have to be one pixel at time)
        self.padding = padding # means giving additional pixels at the boundary of the data.bPixels at the corner are less counted than those in the middle. This means that the pixels don’t get the same amount of weights.
        # weights
        self.weights = np.random.rand(kernel_shape[0], kernel_shape[1], self.in_ch, self.out_ch) - 0.5
        # bias
        self.bias = np.zeros((self.out_ch, 1))

    def forward(self, input):
        s = self.stride
        self.X_shape = input.shape
        _, p = pad2D(input, self.padding, self.weights.shape[:2], s)

        pr1, pr2, pc1, pc2 = p
        fr, fc, in_ch, out_ch = self.weights.shape
        n_ex, in_rows, in_cols, in_ch = input.shape

        # compute the dimensions of the convolution output
        out_rows = int((in_rows + pr1 + pr2 - fr) / s + 1)
        out_cols = int((in_cols + pc1 + pc2 - fc) / s + 1)

        # convert X and W into the appropriate 2D matrices and take their product
        self.X_col, _ = im2col(input, self.weights.shape, p, s)
        W_col = self.weights.transpose(3, 2, 0, 1).reshape(out_ch, -1)

        output_data = (W_col @ self.X_col + self.bias).reshape(out_ch, out_rows, out_cols, n_ex).transpose(3, 1, 2, 0)

        return output_data

    def backward(self, erro, learning_rate):

        fr, fc, in_ch, out_ch = self.weights.shape
        p = self.padding
        db = np.sum(erro, axis=(0, 1, 2))
        db = db.reshape(out_ch,)

        dout_reshaped = erro.transpose(1, 2, 3, 0).reshape(out_ch, -1)
        dW = dout_reshaped @ self.X_col.T
        dW = dW.reshape(self.weights.shape)

        W_reshape = self.weights.reshape(out_ch, -1)
        dX_col = W_reshape.T @ dout_reshaped
        input_error = col2im(dX_col, self.X_shape, self.weights.shape, (p, p, p, p), self.stride)

        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db

        return input_error

class Pooling2D(Layer):

    """
    Pooling is the process of merging. So it’s basically for the purpose of reducing the size of the data.
    """
    
    def __init__(self, size = 2, stride = 1):
        self.size = size
        self.stride = stride

    def pool(X_col):  # self?
        raise NotImplementedError

    def dpool(dX_col, dout_col, pool_cache):  # self?
        raise NotImplementedError

    def forward(self, input):
        self.X_shape = input.shape
        n, h, w, d = input.shape
        h_out = (h - self.size) / self.stride + 1
        w_out = (w - self.size) / self.stride + 1

        if not w_out.is_integer() or not h_out.is_integer():
            raise Exception('Invalid output dimension')

        h_out, w_out = int(h_out), int(w_out)
        X_reshaped = input.transpose(0,3,1,2)
        X_reshaped = X_reshaped.reshape(n * d, h, w, 1)
        self.X_col, _ = im2col(X_reshaped, (self.size, self.size, d, d), pad=0, stride=self.stride)

        out, self.max_idx = self.pool(self.X_col)

        out = out.reshape(d, h_out, w_out, n)
        out = out.transpose(3, 1, 2, 0)

        return out

    def backward(self, erro, learning_rate):
        n, w, h, d = self.X_shape

        dX_col = np.zeros_like(self.X_col)
        dout_col = erro.transpose(1, 2, 3, 0).ravel()

        dX = self.dpool(dX_col, dout_col, self.max_idx)
        dX = col2im(dX, (n * d, h, w, 1), (self.size, self.size, d, d), pad=0, stride=self.stride)
        dX = dX.reshape(self.X_shape)

        return dX

class MaxPooling2D(Pooling2D):

    """
    Max Pooling is a downsampling strategy in Convolutional Neural Networks.
    """
    
    def pool(self, X_col):
        print(X_col.shape)
        max_idx = np.argmax(X_col, axis = 0) # Returns the indices of the maximum values along an axis = array de index
        out = X_col[max_idx, range(max_idx.size)] # vai buscar o index ate aquele tamanho
        return out, max_idx

    def dpool(self, dX_col, dout_col, pool_cache):
        dX_col[pool_cache, range(dout_col.size)] = dout_col
        return dX_col