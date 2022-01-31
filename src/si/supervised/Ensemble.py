from.Model import Model
import numpy as np

def majority(values):
    return max(set(values), key=values.count)#melhor valor


def average(values):
    return sum(values)/len(values)#ou a media


class Ensemble(Model):
    def __init__(self, models, fvote, score):
        super(Ensemble, self).__init__()
        self.models = models#lista de modelos
        self.fvote = fvote#majority ou average
        self.score = score

    def fit(self, dataset):
        self.dataset = dataset
        for model in self.models:
            model.fit(dataset)
        self.is_fitted = True

    def predict(self, x):
        assert self.is_fitted, 'Model not fitted'
        preds = [model.predict(x) for model in self.models]#return de uma lista com o predict the x a partir dos diferentes modelos na lista
        vote = self.fvote(preds)#vai realizar o majority ou o average dos valores do predict escolhendo o melhor
        return vote

    def cost(self, X=None, Y=None):
        X = X if X is not None else self.dataset.X
        Y = Y if Y is not None else self.dataset.Y

        Y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=X.T)#arr e masked
        #Y_pred e o nosso masked
        return self.score(Y, Y_pred)#accuracy_score
