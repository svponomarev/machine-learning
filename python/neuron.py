import numpy as np
import random
"""
neuron.py
Общий класс нейрона + функции вычисления градиента, аналитически и численно.
При наследовании необходимо реализовать функцию стоимости, активационную функцию и производную активационной функции
(для аналитического расчета градиента).
"""
class Neuron:
    def __init__(self, w, cost_function, activation_function, activation_function_derivative):
        """
        w - вектор весов нейрона размерностью (n + 1, 1), где n - число признаков, w[0, 0] - смещение
        activation_function - активационная функция нейрона
        activation_function_derivative - производная активационной функции нейрона (для градиентного спуска)
        cost_function - функция стоимости
        """
        
        assert w.shape[1] == 1, "Incorrect weights shape"
        self.w = w
        self.cost_function = cost_function
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        
    def forward_pass(self, single_input):
        """
        Активационная функция нейрона для единичного примера
        single_input - вектор размерностью (n + 1, 1), где n - число признаков
        """
        sum = 0
        for i in range(self.w.size):
            sum += float(self.w[i] * single_input[i])
        return self.activation_function(sum)
    
    def summatory(self, input_matrix):
        """
        Сумматорная функция нейрона. Суммирует произведения весов и значений признаков для набора примеров.
        input_matrix - размерностью (m, n + 1), где m - число примеров, n - число признаков.
        Возвращает вектор сумм для набора примеров размерностью (m, 1).
        """
        return input_matrix.dot(self.w)
        
    def activation(self, summatory_activation):
        """
        Активационная функция нейрона для набора примеров. Получает на вход результат работы сумматорной функции,
        вектор summatory_activation размерностью (m, 1), где m - число примеров.
        Возвращает вектор (m, 1) со значениями активации для всех примеров.
        """
        return self.activation_function(summatory_activation)
        
    def vectorized_forward_pass(self, input_matrix):
        """
        Векторизованная активационная функция для набора примеров.
        input_matrix - матрица размерностью (m, n + 1), где m - число примеров, n - число признаков.
        Возвращает вектор (m, 1) со значениями активации для всех примеров.
        """
        return self.activation(self.summatory(input_matrix))
    def GD(self, X, y, batch_size, learning_rate = 12.0, eps = 1e-18, max_steps = 10000):
        """
        Внешний цикл алгоритма градиентного спуска.
        X - матрица входных активаций (m, n + 1)
        y - вектор правильных ответов (m, 1)
        learning_rate - константа скорости обучения
        batch_size - размер обучающей выборки
        eps - критерий остановки - величина разности между значениями целевой функции для соседних итераций
        max_steps - критерий остановки №2 - максимально возможное число итераций
        Метод возвращает 1, если сработал первый критерий остановки (алгоритм сошелся), и 0,
        если сработал второй критерий (алгоритм не успел сойтись)
        """
        for step in range(max_steps):
            idx = np.random.choice(len(X), batch_size, replace = False)
            if self.update_mini_batch(X[idx], y[idx], learning_rate, eps):
                return 1
        return 0
            
    def update_mini_batch(self, X, y, learning_rate, eps):
        """
        Внутренняя часть GD - рассчитывает градиент и обновляет веса нейрона.
        X - матрица размера (batch_size, n + 1)
        y - вектор правильных ответов размера (batch_size, 1)
        learning_rate - константа скорости обучения
        eps - критерий остановки номер один: если разница между значением целевой функции 
        до и после обновления весов меньше eps - алгоритм останавливается.
        Возвращает 1 при срабатывании критерия, 0 в остальных случаях
        """
        J = self.cost_function(self, X, y)
        grad = compute_grad_numerically_2(self, X, y, self.cost_function)
        self.w -= learning_rate * grad
        return abs(J - self.cost_function(self, X, y)) < eps
    
    def step_by_step_costs(self, X, y, batch_size, costs, learning_rate = 12.0, eps = 1e-18, max_steps = 10000):
        """
        Реализация градиентного спуска с сохранением значения функции стоимости на каждой итерации в список costs
        X - матрица входных активаций (m, n + 1)
        y - вектор правильных ответов (m, 1)
        learning_rate - константа скорости обучения
        batch_size - размер обучающей выборки
        eps - критерий остановки - величина разности между значениями целевой функции для соседних итераций
        max_steps - критерий остановки №2 - максимально возможное число итераций
        Метод возвращает 1, если сработал первый критерий остановки (алгоритм сошелся), и 0,
        если сработал второй критерий (алгоритм не успел сойтись)
        """
        for step in range(max_steps):
            idx = np.random.choice(len(X), batch_size, replace = False)
            costs.append(self.cost_function(self, X, y))
            if self.update_mini_batch(X[idx], y[idx], learning_rate, eps):
                return 1
        return 0
        
def J_derivative(y, y_hat):
    """
    Вычисляет вектор частных производных для каждого наблюдения
    """
    assert y_hat.shape == y.shape and y_hat.shape[1] == 1, "Incorrect shapes"
    return (y_hat - y) / len(y)
    
def compute_grad_analitically(neuron, X, y, J_prime = J_derivative):
    """
    Аналитическая производная функции стоимости.
    neuron - объект класса Neuron
    X - вертикальная матрица входов формы (m, n + 1), на которой считается сумма квадратов отклонений
    y - правильные ответы для примеров из матрицы X
    J_prime - функция, считающая производные целевой функции по ответам
    Возвращает вектор размера (m, 1)
    """
    z = neuron.summatory(X) # z - вектор результатов сумматорной функции нейрона на разных примерах
    y_hat = neuron.activation(z) # значение активационной функции для всех примеров
    
    dy_dyhat = J_prime(y, y_hat) # производная функции стоимости
    dyhat_dz = neuron.activation_function_derivative(z) # производная активационной функции
    
    dz_dw = X # производная от аргумента активационной функции (x*w)' = x
    
    grad = ((dy_dyhat * dyhat_dz).T).dot(dz_dw) # вычисление сложной производной
    
    grad = grad.T # переход от строки к столбцу
    
    return grad
    
def compute_grad_numerically_2(neuron, X, y, J, eps=1e-5):
    """
    Численная производная функции стоимости
    neuron - объект класса Neuron с вертикальным вектором весов w,
    X - вертикальная матрица входов формы (n, m), на которой считается сумма квадратов отклонений,
    y - правильные ответы для тестовой выборки X,
    J - целевая функция, градиент которой мы хотим получить,
    eps - размер delta w (малого изменения весов).
    Возвращает вектор размера (m, 1)
    """
    w_0 = neuron.w
    num_grad2 = np.zeros(w_0.shape)
    for i in range(len(w_0)):
        old_w = w_0[i].copy()
        neuron.w[i] -= eps
        Jminus = J(neuron, X, y)
        neuron.w[i] = old_w + eps
        Jplus = J(neuron, X, y)
        num_grad2[i] = (Jplus - Jminus)/(2*eps) # grad(f(x)) = (f(x + dx) - f(x - dx))/(2*dx)
        neuron.w[i] = old_w        
    assert np.allclose(neuron.w, w_0), "МЫ ИСПОРТИЛИ НЕЙРОНУ ВЕСА - 2"
    return num_grad2