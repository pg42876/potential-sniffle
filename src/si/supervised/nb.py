import numpy as np
from src.si.supervised.Model import Model
from src.si.util.Metrics import accuracy_score

class Naive_bayes(Model):

    def __init__(self):
        super(Naive_bayes, self).__init__()

    def fit(self, dataset):
        
        """
        P (class | data) = P (data | class) * P (class) / P (data)
        P (y | x) = P (x | y) * P (y) / P (x)
        posterior = likelihood * prior / evidence
        """
        
        self.dataset = dataset
        self.classes = np.unique(dataset.Y)
        n_rows = dataset.X.shape[0]

        X_byclass = np.array([dataset.X[dataset.Y == classe] for classe in self.classes])
        self.prob_class = np.array([len(X_class) / n_rows for X_class in X_byclass])  # probabilidade da classe

        self.mean = np.asarray([np.mean(rows_classe, axis = 0) for rows_classe in X_byclass])
        self.var = np.asarray([np.var(rows_classe, axis = 0) for rows_classe in X_byclass])

        self.is_fitted = True

    def gaussian_proba(self, x):
        exponent = np.exp(-((x - self.mean) ** 2 / (2 * self.var ** 2)))
        prob = (1 / (np.sqrt(2 * np.pi) * self.var)) * exponent
        return prob

    def predict(self, x):
        assert self.is_fitted, 'Model must be fitted before prediction'
        probabilities = np.prod(self.gaussian_proba(x), axis = 1) * self.prob_class
        idx = np.argmax(probabilities)
        prediction = self.classes[idx]
        return prediction

    def cost(self):
        y_pred = np.ma.apply_along_axis(self.predict, axis = 0, arr = self.dataset.X.T)
        return accuracy_score(self.dataset.Y, y_pred)