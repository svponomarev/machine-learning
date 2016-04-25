from neuron import * 
import cost_func as cf
import numpy as np
"""
logistic_neuron.py
Класс логистического (сигмоидального) нейрона.
"""
def sigmoid(x):
    # Логистическая (сигмоидальная) активационная функция
    return 1 / (1 + np.exp(-x))
def sigmoid_prime(x):
    # Производная сигмоидальной активационной функции
    return sigmoid(x) * (1 - sigmoid(x))

class LogisticNeuron(Neuron):
    def __init__(self, w, cost_function = cf.J_logarifmic, activation_function = sigmoid, activation_function_derivative = sigmoid_prime):
        super(LogisticNeuron, self).__init__(w, cf.J_logarifmic, sigmoid, sigmoid_prime)

def create_logistic_neuron(n, cost_function = cf.J_logarifmic, activation_function = sigmoid, activation_function_derivative = sigmoid_prime):
    """
    Создаем логистический нейрон со случайными весами и единичным смещением.
    """
    w = np.random.random((n + 1, 1))
    w[0] = 1
    return LogisticNeuron(w, cost_function, activation_function, activation_function_derivative)