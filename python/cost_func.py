import numpy as np
"""
cost_func.py
Функции стоимости (целевые функции) для решения задач регрессии и классификации.
"""
def J_quadratic(neuron, X, y):
    assert y.shape[1] == 1, "Incorrect shape"
    prediction = neuron.vectorized_forward_pass(X)
    return 0.5 * ((prediction - y).T.dot(prediction - y)) / X.shape[0]
    
def J_logarifmic(neuron, X, y):
    assert y.shape[1] == 1, "Incorrect shape"
    prediction = neuron.vectorized_forward_pass(X)
    return - ((y.T).dot(np.log(prediction)) + (1 - y.T).dot(np.log(1 - prediction))) / X.shape[0]