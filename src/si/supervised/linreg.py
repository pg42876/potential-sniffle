import numpy as np
from src.si.data.Dataset import Dataset
from src.si.supervised.Model import Model
from src.si.util.Metrics import mse

class LinearRegression(Model):

    def __init__(self, gd = False, epochs = 1000, lr = 0.001):
        super(LinearRegression, self).__init__()
        self.gd = gd
        self.theta = None
        self.num_inter = epochs
        self.lr = lr

    def fit (self, dataset):
        X, Y = dataset.getXy()
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.X = X
        self.Y = Y

        # closed from of GD
        if self.gd:
            self.train_gd(X, Y)
        else:
             self.train_closed(X, Y)
        self.is_fitted = True

    def train_closed(self, x, y):
        self.theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    
    def train_gd(self, x, y):
        m = x.shape[0]
        n = x.shape[1]
        self.history = {} # dicionário com o histórico dos thetas e custo por epoch
        self.theta = np.zeros(n)
        for epoch in range(self.num_inter):
            grad = 1 / m * (x.dot(self.theta) - y).dot(x)
            self.theta -= self.lr * grad
            self.history[epoch] = [self.theta[:], self.cost()]
    
    def predcit(self, x):
        assert self.is_fitted, 'Model must be fitted before predicting'
        _x = np.hstack(([1], x))
        return np.dot(self.theta, _x)

    def cost(self):
        y_pred = np.dot(self.X, self.theta)
        return mse(self.Y, y_pred) / 2

class LinearRegressionReg(LinearRegression):
    
    def __init__(self, gd = False, epochs = 1000, lr = 0.001, lbd = 1):
        super(LinearRegressionReg, self).__init__()
        self.lbd = lbd
        self.gd = gd
        self.num_inter = epochs

    def train_closed(self, x, y):
        n = x.shape[1]
        identify = np.eye(n)
        identify[0, 0] = 0 # mudar o primeiro elemento para 0 para não dar biased
        self.theta = np.linalg.inv(x.T.dot(x) + self.lbd * identify).dot(x.T).dot(y)
        self.is_fitted = True

    def train(self, x, y):
        m = x.shape[0]
        n = x.shape[1]
        self.history = {} # dicionário com o histórico dos thetas por epoch
        self.theta = np.zeros(n)
        lbds = np.full(m, self. lbd)
        lbds[0] = 0
        for epoch in range(self.num_inter):
            grad = 1 / m * (x.dot(self.theta) - y).dot(x)
            self.theta -= (self.lr / m) * (lbds + grad)
            self.history[epoch] = [self.theta[:], self.cost()]
    
    def predict(self, x):
        assert self.is_fited
        _x = np.hstack(([1], x))
        return np.dot(self.theta, _x)

    def cost(self):
        y_pred = np.dot(self.X, self.theta)
        return mse(self.Y, y_pred) / 2