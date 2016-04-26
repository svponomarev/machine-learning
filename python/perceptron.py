import numpy as np
import random
"""
perceptron.py
Класс перцептрона + функция для визуализации динамики обучения (график количества ошибок от числа итераций).
"""
random.seed(42)

class Perceptron:
    def __init__(self, w):
        """ 
        Инициализация перцептрона
        w - вектор весов размерностью (n + 1, 1), n - число признаков
        """
        self.w = w

    def forward_pass(self, single_input):
        """
        Расчет ответа перцептрона при предъявлении одного примера
        single_input - вектор примера размерностью (n + 1, 1)
        Метод возвращает True/False 
        """
        result = 0
        for i in range(0, len(self.w)):
            result += self.w[i] * single_input[i]
                
        if result > 0:
            return 1
        else:
            return 0
    def vectorized_forward_pass(self, input_matrix):
        """
        Расчет вектора ответов перцептрона при предъявлении набора примеров
        input_matrix - матрица размерностью (m, n + 1), где m - число наблюдений,
        n - число признаков
        Метод возвращает вектор ответов перцептрона (True/False) размерностью (m, 1)
        """
        result = np.dot(input_matrix, self.w)
        result = (result > 0)   
        return result
    def train_on_single_example(self, example, y):
        """
        Принимает вектор активации входов example (n + 1, 1)
        и ожидаемый ответ y (0 / 1). Обновляет значения весов в случае ошибки.
        Возвращает размер ошибки (0 или 1).
        """
        response = self.vectorized_forward_pass(example.T)
        if response != y:
            self.w += (y - response) * example
            return 1
        else:
            return 0

            
    def train_until_convergence(self, input_matrix, y, max_steps = 1e08):
        """
        Обучение перцептрона на наборе примеров.
        input_matrix - матрица примеров (m, n + 1), где
        m - число примеров, n - число признаков
        y - вектор с правильными ответами (True/False) размерностью (m, 1)
        max_steps - максимальное количество итераций алгоритма
        """
        i = 0
        errors = 1
        while errors and i < max_steps:
            i += 1
            errors = 0
            for example, answer in zip(input_matrix, y):
                example = example.reshape((example.size, 1))
                print(example)
                error = self.train_on_single_example(example, answer)
                errors += int(error)
        if errors == 0:
            return 1
        else:
            return 0
                
def create_perceptron(n):
    """
    Создаем перцептрон со случайными весами и единичным смещением.
    """
    w = np.random.random((n + 1, 1))
    w[0] = 1
    return Perceptron(w)
def step_by_step_errors(p, input_matrix, y, max_steps = 1e5):
    """
    Обучаем перцептрон последовательно на каждой строчке данных
    и запоминаем количество ошибок на каждой итерации. Возвращаем их в виде списка.
    """
    def count_errors():
        return np.abs(p.vectorized_forward_pass(input_matrix).astype(np.int) - y).sum()
        
    errors_list = [count_errors()]
    i = 0
    errors = 1
    while errors and i < max_steps:
        i += 1
        errors = 0
        for example, answer in zip(input_matrix, y):
            example = example.reshape((example.size, 1))
            error = p.train_on_single_example(example, answer)
            errors += int(error)           
            errors_list.append(count_errors())
    return errors_list
    