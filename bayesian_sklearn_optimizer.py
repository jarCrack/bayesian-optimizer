import GPy
import GPyOpt
import numpy as np
from sklearn import svm
from numpy.random import seed
from matplotlib.pyplot import *
from sklearn.cross_validation import train_test_split

seed(12345)


class BayesianSkLearnOptimizer:

    def evaluate(self,hyperparams):
        print(hyperparams)
        trained_model=self.model()
        # self.model=self.model()
        return 1.

    def __init__(self, sklearn_model, params_to_tune, training_and_test_data, custom_validation=None):
        """
        Initialization of the Bayesian sklearn model optimizer.

        :param sklearn_model:
        :param params_to_tune: Hyperparameters to tune, given in the format of GPyOpt
        :param training_and_test_data:
        :param custom_validation: Custom validation function, if your need some nifty cross validation stuff :)
        """
        self.hyperparameters = params_to_tune
        self.model = sklearn_model
        validation_method = custom_validation if callable(custom_validation) else lambda x:self.evaluate(x)

        problem = GPyOpt.methods.BayesianOptimization(f=validation_method, domain=self.hyperparameters,
                                                      acquisition_type="LCB", acquisition_weight=0.1)

        print(hyperparameters)

if __name__ == "__main__":
    sklearn_model = svm.SVR

    hyperparameters = [{'name': 'C', 'type': 'continuous', 'domain': (0., 7.)},
                       {'name': 'epsilon', 'type': 'continuous', 'domain': (-12., -2.)},
                       {'name': 'gamma', 'type': 'continuous', 'domain': (-12., -2.)}]

    data = GPy.util.datasets.olympic_marathon_men()
    X = data['X']
    y = data['Y']

    data = train_test_split(X, y, test_size=0.33, random_state=42)

    optimizer = BayesianSkLearnOptimizer(sklearn_model, hyperparameters, data)
