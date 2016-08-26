import GPy
import GPyOpt
import numpy as np
from sklearn import svm
from numpy.random import seed
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
seed(12345)


class BayesianSkLearnOptimizer:

    def evaluate(self,hyperparams):
        print(hyperparams)
        hyps={c:int(hyperparams[0,i]) for i,c in enumerate(self.hyp_names)}
        self.trained_model=self.model(**hyps)
        self.trained_model.fit(self.X_train,self.y_train)

        plt.plot(self.X_test, self.y_test, 'x')
        plt.plot(self.X_test, self.trained_model.predict(self.X_test), '.')
        plt.show()

        err=mean_absolute_error(self.y_test,self.trained_model.predict(self.X_test))
        print(err)
        return err
        #return self.trained_model.score(self.X_test,self.y_test)
    def __init__(self, sklearn_model, params_to_tune, training_and_test_data, custom_validation=None,iteration_callback=None):
        """
        Initialization of the Bayesian sklearn model optimizer.

        :param sklearn_model:
        :param params_to_tune: Hyperparameters to tune, given in the format of GPyOpt
        :param training_and_test_data:
        :param custom_validation: Custom validation function, if your need some nifty cross validation stuff :)
        """

        self.hyperparameters = params_to_tune
        self.hyp_names=[x["name"] for x in self.hyperparameters]
        self.X_train, self.X_test, self.y_train, self.y_test = training_and_test_data
        self.model = sklearn_model
        self.iteration_callback=iteration_callback

        validation_method = custom_validation if callable(custom_validation) else lambda x:self.evaluate(x)

        self.problem = GPyOpt.methods.BayesianOptimization(f=validation_method, domain=self.hyperparameters,
                                                      acquisition_type="LCB", acquisition_weight=0.1)

        print(hyperparameters)
    def optimize_model(self):
        self.problem.run_optimization(max_iter=10)
        return self.trained_model

if __name__ == "__main__":
    sklearn_model = svm.SVR

    hyperparameters = [{'name': 'C', 'type': 'continuous', 'domain': (0., 7.)},
                       {'name': 'epsilon', 'type': 'continuous', 'domain': (-12., -2.)},
                       {'name': 'gamma', 'type': 'continuous', 'domain': (-12., -2.)}]

    from sklearn.ensemble import RandomForestRegressor
    sklearn_model_2=RandomForestRegressor

    hyperparameters = [{'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 100)},
                       {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 100)}]
    data = GPy.util.datasets.olympic_marathon_men()
    X = data['X']
    y = data['Y']

    data = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train, X_test, y_train, y_test = data

    optimizer = BayesianSkLearnOptimizer(sklearn_model_2, hyperparameters, data)
    model=optimizer.optimize_model()

    plt.plot(X_test, y_test, 'x')
    plt.plot(X_test, model.predict(X_test), '.')
    plt.show()