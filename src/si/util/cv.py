import numpy as np  
import itertools
from numpy.core.numeric import tensordot
from numpy.lib.arraypad import pad
from si import data
from ..util import train_test_split

class CrossValidation:

    def __init__(self, model, dataset, score = None, **kwargs):
        self.model = model
        self.dataset = dataset
        self.score = score
        self.cv = kwargs.get('cv', 3)
        self.split = kwargs.get('split', 0.8)
        self.train_score = None
        self.test_score = None
        self.ds = None

    def run(self):
        train_score = []
        test_score = []
        ds = []
        for _ in range(self.cv): # o _ serve para representação do valor não variável (não guarda)
            train, test = train_test_split(self.dataset, self.split)
            ds.append((train, test))
            self.model.fit(train)
            train_score.append(self.model.cost())
            test_score.append(self.model.cost(test.X, test.Y))
        self.train_score = train_score # guarda os dados que estavam escritos anteriormente de forma a preservar os mesmos
        self.test_score = test_score # guarda os dados que estavam escritos anteriormente de forma a preservar os mesmos
        self.ds = ds
        return train_score, test_score

    def toDataFrame(self):
        import pandas as pd
        assert self.train_score and self.test_score, 'Need to run function'
        return pd.DataFrame({'Train Scores:' : self.train_score, 'Test Scores:' : self.test_score})


class GridSearchCV:

    def __init__(self, model, dataset, parameters, **kwargs):
        self.model = model
        self.dataset = dataset
        hasparam = (hasattr(self.model, param) for param in parameters)
        # vai verificar se os parâmetros que estamos a verificar são os parâmetros do modelo (verifica se as chaves são ou não atributos; quando não for atributo do modelo dá erro)
        if np.all(hasparam): # se todos os atributos foram True
            self.parameters = parameters # são parâmetros
        else: # se não forem
            index = hasparam.index(False) # verifica que é False
            keys = list(parameters.keys()) # devolve as keys em forma de lista
            raise ValueError(f'Wrong parameters: {keys[index]}') # dá o erro
        self.kwargs = kwargs
        self.results = None

    def run(self):
        self.results = []
        attrs = list(self.parameters.keys())
        values = list(self.parameters.values())
        for conf in itertools.product(*values):
            for i in range(len(attrs)):
                setattr(self.model, attrs[i], conf[i]) # atribuir os parâmetros do modelo à configuração
            scores = CrossValidationScore(self.model, self.dataset, **self.kwargs).run()
            self.results.append((conf, scores))
        return self.results

    def toDataframe(self):
        import pandas as pd
        assert self.results, "The grid search needs to be ran."
        data = dict()
        for i, k in enumerate(self.parameters.keys()):
            v = []
            for r in self.results:
                v.append(r[0][i])
            data[k] = v
        for i in range(len(self.results[0][1][0])):
            treino = []
            teste = []
            for r in self.results:
                treino.append(r[1][0][i])
                teste.append(r[1][1][i])
            treino = data['Train ' + str(i + 1)] 
            teste = data['Test ' + str(i + 1)] 
        return pd.DataFrame(data)

class CrossValidationScore:

    def __init__(self, model, dataset, score = None, **kwargs):
        self.model = model
        self.dataset = dataset
        self.cv = kwargs.get('cv', 3)
        self.split = kwargs.get('split', 0.8)
        self.train_score = None
        self.test_score = None
        self.ds = None
        self.score = score

    def run(self):
        train_score = []
        test_score = []
        ds = [] # guardar datasets
        true_Y, pred_Y = [], [] 
        for _ in range(self.cv): # o _ serve para representação do valor não variável (não guarda)
            train, test = train_test_split(self.dataset, self.split)
            ds.append((train, test))
            self.model.fit(train)
            if not self.score:
                train_score.append(self.model.cost())
                test_score.append(self.model.cost(test.X, test.Y))
                pred_Y.extend(list(self.model.predict(test.X))) 
            else:
                y_train = np.ma.apply_along_axis(self.model.predict, axis = 0, arr = train.X.T)
                train_score.append(self.score(train.Y, y_train))
                y_test = np.ma.apply_along_axis(self.model.predict, axis = 0, arr = train.X.T)
                test_score.append(self.score(test.Y, y_test))
                pred_Y.extend(list(y_test)) 
            true_Y.extend(list(test.Y)) 
        self.train_score = train_score # guarda os dados que estavam escritos anteriormente de forma a preservar os mesmos
        self.test_score = test_score # guarda os dados que estavam escritos anteriormente de forma a preservar os mesmos
        self.ds = ds
        self.true_Y = np.array(true_Y) 
        self.pred_Y = np.array(pred_Y) 
        return train_score, test_score

    def toDataFrame(self):
        import pandas as pd
        assert self.train_score and self.test_score, 'Need to run function'
        return pd.DataFrame({'Train Scores:' : self.train_score, 'Test Scores:' : self.test_score})