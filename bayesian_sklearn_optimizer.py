import GPy
import GPyOpt
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from itertools import product
import pprint
import copy

np.random.seed(12345)


class BayesianSkLearnOptimizer:

    def evaluate(self, hyperparams):
        hyps = {c["name"]: np.exp(hyperparams[0, i]) for i, c in enumerate(self.numeric_hyperparameters) if
                c["type"] is not "categorical"}
        self.trained_model = self.model(**hyps, **self.categories)
        self.trained_model.fit(self.X_train, self.y_train)

        err = mean_absolute_error(self.y_test, self.trained_model.predict(self.X_test))
        return err

    def __init__(self, sklearn_model, params_to_tune, training_and_test_data, custom_validation=None,
                 iteration_callback=None):
        """
        Initialization of the Bayesian sklearn model optimizer.

        :param sklearn_model:
        :param params_to_tune: Hyperparameters to tune, given in the format of GPyOpt
        :param training_and_test_data:
        :param custom_validation: Custom validation function, if your need some nifty cross validation stuff :)
        """

        self.hyperparameters = params_to_tune
        self.numeric_hyperparameters = [c for c in self.hyperparameters if c["type"] is not "categorical"]
        self.categorical_hyperparameters = [c for c in self.hyperparameters if c["type"] is "categorical"]
        self.categories = [c["choice"] for c in self.categorical_hyperparameters]

        self.X_train, self.X_test, self.y_train, self.y_test = training_and_test_data
        self.model = sklearn_model
        self.iteration_callback = iteration_callback

        self.validation_method = custom_validation if callable(custom_validation) else lambda x: self.evaluate(x)

    def optimize_model(self):

        result={}
        best_score=1e3
        best_model=None

        for el in product(*self.categories):
            self.categories = {c["name"]: el[i] for i, c in enumerate(self.categorical_hyperparameters)}
            self.problem = GPyOpt.methods.BayesianOptimization(f=self.validation_method,
                                                               domain=self.numeric_hyperparameters,
                                                               acquisition_type="EI", acquisition_weight=0.1)

            self.problem.run_optimization(max_iter=20)
            self.problem.plot_convergence()

            cur_score=self.problem.fx_opt[0]
            model_copy=copy.copy(self.trained_model)

            result[el]={'score':cur_score,'model:':model_copy}
            if cur_score<best_score:
                best_score=cur_score
                best_model=model_copy

        p = pprint.PrettyPrinter()
        p.pprint(result)

        return best_model


if __name__ == "__main__":
    sklearn_model = svm.SVR

    hyperparameters = [{'name': 'C', 'type': 'continuous', 'domain': (-7., 7.)},
                       {'name': 'epsilon', 'type': 'continuous', 'domain': (-12., -2.)},
                       {'name': 'gamma', 'type': 'continuous', 'domain': (-12., -2.)},
                       {'name': 'tol', 'type': 'continuous', 'domain': (-5, 0)},
                       {'name': 'kernel', 'type': 'categorical', 'choice': ('rbf', 'linear','poly','sigmoid')}]

    sklearn_model2=RandomForestRegressor
    hyperparameters2 = [{'name': 'n_estimators', 'type': 'discrete', 'domain': (10, 100.)},
                        {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 100)},
                        {'name': 'max_features', 'type': 'categorical', 'choice':('sqrt','log2','auto')}]

    data = GPy.util.datasets.olympic_marathon_men()

    X = data['X']
    y = data['Y']

    data = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train, X_test, y_train, y_test = data
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    data = (X_train, X_test, y_train, y_test)

    optimizer = BayesianSkLearnOptimizer(sklearn_model, hyperparameters, data)
    model = optimizer.optimize_model()

    plt.plot(X_test, y_test, 'x')
    plt.plot(X_test, model.predict(X_test), '.')
    plt.show()
