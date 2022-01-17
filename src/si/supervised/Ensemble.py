import numpy as np
from src.si.supervised.Model import Model

def majority(values):
    return max(set(values), key = values.count)

def average(values):
    return sum(values) / len(values)

class Ensemble(Model):

    def __init__(self, models, fvote, score):
        super().__init__()
        self.models = models
        self.fvote = fvote
        self.score = score
    
    def fit(self, dataset): # vai fazer fit dos modelos
        self.dataset = dataset
        for model in self.models:
            model.fit(dataset)
        self.is_fitted = True

    def predict(self, x):
        assert self.is_fitted, 'Model must be fit before predicting'
        preds = [model.predict(x) for model in self.models]
        vote = self.fvote(preds)
        return vote

    def cost(self, X = None, y = None):
        if X is not None:
            X = X 
        else:
            self.dataset.X
        y = y if y is not None else self.dataset.Y
        y_pred = np.ma.apply_along_axis(self.predcit, axis = 0, arr = X.T)
        return y_pred