from neuron import * 
import cost_func as cf
"""
linear_neuron.py
Класс линейного нейрона.
"""
def linear(x):
    #Активационная функция линейного нейрона возвращает просто аргумент
    return x
def linear_prime(x):
    # Производная линейной функции - константа
    return 1

class LinearNeuron(Neuron):
    def __init__(self, w, cost_function = cf.J_quadratic, activation_function = linear, activation_function_derivative = linear_prime):
        super(LinearNeuron, self).__init__(w, cf.J_quadratic, linear, linear_prime)

def create_linear_neuron(n, cost_function = cf.J_quadratic, activation_function = linear, activation_function_derivative = linear_prime):
    """
    Создаем линейный нейрон со случайными весами и единичным смещением.
    """
    w = np.random.random((n + 1, 1))
    w[0] = 1
    return LinearNeuron(w, cost_function, activation_function, activation_function_derivative)