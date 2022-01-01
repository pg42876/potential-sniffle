import numpy as np
import pandas as pd

def accuracy_score(y_true, y_pred):

    """
    Verifica quais são as previsões que são iguais aos valores reais
    """

    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    accuracy = correct / len(y_true)
    return accuracy

def mse(y_true, y_pred, squared = True):
    
    """
    Mean squared error
    """
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    errors = np.average((y_true - y_pred) ** 2, axis = 0)
    if not squared:
        errors = np.sqrt(errors)
    return np.average(errors)

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def cross_entropy(y_true, y_pred):
    return - (y_true * np.log(y_pred)).sum()

def cross_entropy_prime(y_true, y_pred):
    return y_pred - y_true

def r2_score(y_true, y_pred):

    """
    R^2 regression score function.
        R^2 = 1 - SS_res / SS_tot
    where SS_res is the residual sum of squares and SS_tot is the total
    sum of squares.
    
    :param numpy.array y_true : array-like of shape (n_samples,) Ground truth (correct) target values.
    :param numpy.array y_pred : array-like of shape (n_samples,) Estimated target values.
    :returns: score (float) R^2 score.
    """

    # Residual sum of squares
    numerator = ((y_true - y_pred) ** 2).sum(axis = 0)
    # Total sum of squares
    denominator = ((y_true - np.average(y_true, axis = 0)) ** 2).sum(axis = 0)
    # R^2
    score = 1 - numerator / denominator
    return score

class ConfusionMatrix:

    def __init__(self, true_y, pred_y):
        self.true_y = np.array(true_y)
        self.pred_y = np.array(pred_y)
        self.conf = None

    def calc(self):
        self.conf = pd.crosstab(self.true_y, self.pred_y, rownames = ['Actual'], colnames = ['Predicted'], margins = True)

    def toDataframe(self):
        return self.conf

    def __call__(self):
        self.calc()