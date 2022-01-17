import numpy as np
from src.si.supervised.Model import Model
from ..util.Util import sigmoide, add_intersect

class LogisticRegression(Model):

    def __init__(self, gd = False, epochs = 1000, lr = 0.1):
        super(LogisticRegression, self).__init__()
        self.gd = gd
        self.theta = None
        self.num_inter = epochs
        self.lr = lr

    def fit(self, dataset):
        X, Y = dataset.getXy()
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.X = X
        self.Y = Y
        self.train(X, Y)
        self.is_fitted = True

    def train(self, x, y):
        n = x.shape[1]
        self.history = {} # criar um histórico dos thetas e custo por epoch
        self.theta = np.zeros(n)
        for epoch in range(self.num_inter):
            z = np.dot(x, self.theta)
            h = sigmoide(z)
            grad = np.dot(x.T, (h - y)) / y.size
            self.theta -= self.lr * grad
            self.history[epoch] = [self.theta[:], self.cost()]

    def probability(self, x):
        assert self.is_fitted, 'Model must be fitted before predicting'
        _x = np.hstack(([1], x))
        return sigmoide(np.dot(self.theta, _x))

    def predcit(self, x):
        prob = self.probability(x)
        if prob >= 0.5:
            result = 1 
        else:
            result = 0
        return result

    def cost(self, X = None, y = None, theta = None):
        if X is not None:
            X = add_intersect(X)
        else:
            self.X
        if y is not None:
            Y = y 
        else:
            self.Y
        if theta is not None:
            theta = theta 
        else:
            self.theta

        h = sigmoide(np.dot(self.X, self.theta))
        cost = (- self.Y * np.log(h) - (1 - self.Y) * np.log(1 - h))
        res = np.sum(cost) / self.X.shape[0]
        return res

class LogisticRegressionReg(LogisticRegression):
    
    def __init__(self, gd = False, epochs = 1000, lr = 0.1, lbd = 1):
        super(LogisticRegressionReg, self).__init__()
        self.lbd = lbd
        self.gd = gd
        self.num_inter = epochs
        self.lr = lr

    def train(self, x, y):
        n = x.shape[1]
        m = x.shape[0]
        self.history = {} # criar um histórico dos thetas e custo por epoch
        self.theta = np.zeros(n)
        for epoch in range(self.epochs):
            z = np.dot(x, self.theta)
            h = sigmoide(z)
            grad = np.dot(x.T, (h - y)) / y.size
            grad[1:] = grad[1:] + (self.lbd / m) * self.theta[1:]
            self.theta -= self.lr * grad
            self.history[epoch] = [self.theta[:], self.cost()]

    def cost(self, X = None, y = None, theta = None):
        if X is not None:
            X = add_intersect(X)
        else:
            self.X
        if y is not None:
            Y = y 
        else:
            self.Y
        if theta is not None:
            theta = theta 
        else:
            self.theta

        m = X.shape[0]
        p = sigmoide(np.dot(X, theta))
        cost = (-Y * np.log(p) - (1 - Y) * np.log(1 - p))
        reg = np.dot(theta[1:], theta[1:]) * self.lbd / (2 * m)
        res = (np.sum(cost) / m) + reg
        return res 