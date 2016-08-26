# --- Load GPyOpt

import GPy
import GPyOpt
import numpy as np

def myf(x):
    return (2*x)**2

bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (-1,1)}]
myProblem = GPyOpt.methods.BayesianOptimization(myf,bounds)

max_iter=15
myProblem.run_optimization(max_iter)
myProblem.plot_acquisition()
