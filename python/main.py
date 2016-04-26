import numpy as np
import matplotlib.pyplot as plt
import perceptron as pc
import linear_neuron as linn
import logistic_neuron as logn
import cost_func as cf
"""
main.py
Главный модуль - содержит набор операций для тестирования алгоритмов обучения нейронов.
"""
def plot_decision_line(coefs, color = "black", label = "decision boundary", ls = "--"):
    """
    Рисует разделяющую прямую, соответствующую весам, переданным в coefs,
    где coefs  - ndarray формы (3, 1) [b theta_0 theta_1]
    """
    bias = coefs[0, 0]
    w = coefs[1:3, 0]
    xx = np.linspace(*plt.xlim())
    yy = (-1/w[1])*(w[0] * xx + bias)
    plt.plot(xx, yy, color=color, linestyle = ls, linewidth=2, label = label)

def plot_line(coefs, color = "black", label = "decision boundary", ls = "--"):
    """
    Рисует прямую, аппроксимирующую точки на графике, с коэффициентами coefs
    coefs - вектор формы (2, 1) [b theta_1]
    """
    bias = coefs[0, 0]
    w = coefs[1, 0]
    xx = np.linspace(*plt.xlim())
    yy = w * xx + bias
    plt.plot(xx, yy, color=color, linestyle = ls, linewidth=2, label = label)

def formatted_output(neuron, X, y, bs, lr, eps, ms, title, color):
    """
    Форматированный вывод для результатов классификации нейронами
    """
    print(title, ":")
    print("Equation before learning: %.2f*x1 + %.2f*x2 + %.2f = 0" % (neuron.w[2], neuron.w[1], neuron.w[0]))
    print("Cost function before learning: %.2f" % neuron.cost_function(neuron, X, y))
    print("Is logistic neuron converge =", neuron.GD(X, y, bs, lr, eps, ms))
    print("Equation after learning: %.2f*x1 + %.2f*x2 + %.2f = 0" % (neuron.w[2], neuron.w[1], neuron.w[0]))
    print("Cost function after learning: %.2f" % neuron.cost_function(neuron, X, y))
    plot_decision_line(neuron.w, color, title, "--")
    
# 1. Загрузка и визуализация входных данных для регрессии
fig = plt.figure(1)
fig.canvas.set_window_title("Profit of city estimation")
data_r = np.loadtxt("..\\data\\regression_data.csv", delimiter = ",")
y = data_r[:, 1][:, np.newaxis]
b = np.ones(y.shape[0])[:, np.newaxis]
x = data_r[:, 0][:, np.newaxis]
X = np.hstack((b, x))
plt.scatter(data_r[:, 0], data_r[:, 1])
plt.xlabel("Population of city in 10,000s")
plt.ylabel("Profit in $10,000s")

# Обучение линейного нейрона
n = data_r.shape[1] - 1 # n - число признаков
m = data_r.shape[0] # m - число примеров
batch_size = m # batch_size - размер одной "порции" примеров для итерации градиентного спуска
learning_rate = 0.01 # learning_rate - коэффициент скорости обучения
eps = 1e-5 # eps - минимальная разность в функции стоимости, при которой будет продолжена работа алгоритма
max_steps = 1000 # max_steps - максимальное число итераций градиентного спуска

neuron_1 = linn.create_linear_neuron(n)
print("Equation before learning: y = %.2f*x + %.2f" % (neuron_1.w[1], neuron_1.w[0]))
print("Cost function before learning: %.2f" % neuron_1.cost_function(neuron_1, X, y))

print("Is gradient descent converge =", neuron_1.GD(X, y, batch_size, learning_rate, eps, max_steps))

print("Equation after learning: y = %.2f*x + %.2f" % (neuron_1.w[1], neuron_1.w[0]))
print("Cost function after learning: %.2f" % neuron_1.cost_function(neuron_1, X, y))

plot_line(neuron_1.w, "black", "Linear regression", "-")
plt.legend(loc = "lower right")
plt.show()

# Вывод изменения стоимости для линейного нейрона
fig2 = plt.figure(2)
fig2.canvas.set_window_title("Dynamics of cost function")
neuron_1t = linn.create_linear_neuron(n)
costs_1 = []
neuron_1t.step_by_step_costs(X, y, batch_size, costs_1, learning_rate, eps, max_steps)
plt.plot(np.squeeze(costs_1))
plt.xlabel("Algorithm step number")
plt.ylabel("Cost function")
plt.show()

# 2. Загрузка и визуализация входных данных для классификации
fig3 = plt.figure(3)
fig3.canvas.set_window_title("Apple/Pears classification")
data_c = np.loadtxt("..\\data\\classification_data.csv", delimiter = ",")
pears = data_c[:, 2] == 1 # В pears находится маска data, для которой в третьем столбце значения равны 1
apples = np.logical_not(pears) # В apple находится маска data, для которой в третьем столбце значения не равны 1
plt.scatter(data_c[apples][:, 0], data_c[apples][:, 1], color = "red", label = "apples")
plt.scatter(data_c[pears][:, 0], data_c[pears][:, 1], color = "blue", label = "pears")
plt.xlabel("Yellowness")
plt.ylabel("Symmetry")

# Обучение перцептрона
n = data_c.shape[1] - 1 # n - число признаков
m = data_c.shape[0] # m - число примеров
y = data_c[:, 2][:,np.newaxis]
b = np.ones((m, 1))
X = np.hstack((b, data_c[:, 0:2])) # X размера (m, n + 1)

p = pc.create_perceptron(n)
print("Equation before learning: %.2f*x1 + %.2f*x2 + %.2f = 0" % (p.w[2], p.w[1], p.w[0]))

print("Is perceptron converge =", p.train_until_convergence(X, y))

print("Equation after learning: %.2f*x1 + %.2f*x2 + %.2f = 0" % (p.w[2], p.w[1], p.w[0]))

plot_decision_line(p.w, "black", "Perceptron", "--")

# Обучение сигмоидальных нейронов (квадратичная и логарифмическая функции стоимости)
neuron_2 = logn.create_logistic_neuron(n, cf.J_quadratic)
neuron_3 = logn.create_logistic_neuron(n, cf.J_logarifmic)
batch_size = 40 # batch_size - размер одной "порции" примеров для итерации градиентного спуска
learning_rate = 5.0 # learning_rate - коэффициент скорости обучения
eps = 1e-6 # eps - минимальная разность в функции стоимости, при которой будет продолжена работа алгоритма
max_steps = 10000 # max_steps - максимальное число итераций градиентного спуска

formatted_output(neuron_2, X, y, batch_size, learning_rate, eps, max_steps, "Logistic-QUADRATIC", "blue")
formatted_output(neuron_3, X, y, batch_size, learning_rate, eps, max_steps, "Logistic-LOGARIFMIC", "green")
plt.legend(loc = "lower right")
plt.show()

# 3. Подсчет числа ошибок для перцептрона
fig4 = plt.figure(4)
fig4.canvas.set_window_title("Number of errors on each iteration")
p2 = pc.create_perceptron(n)
errors_list = pc.step_by_step_errors(p2, X, y)
print("Number of iterations:", len(errors_list))
print("Final number of errors:", errors_list[len(errors_list) - 1])
plt.plot(errors_list)
plt.ylabel("Number of errors")
plt.xlabel("Algorithm step number")

plt.show()

# 4. Вывод изменения функции стоимости для нейронов
# Квадратичная функция стоимости
fig5 = plt.figure(5)
fig5.canvas.set_window_title("Dynamics of cost function (QUADRATIC)")
neuron_2t = logn.create_logistic_neuron(n, cf.J_quadratic)
costs_2 = []
neuron_2t.step_by_step_costs(X, y, batch_size, costs_2, learning_rate, eps, max_steps)
plt.plot(np.squeeze(costs_2))
plt.xlabel("Algorithm step number")
plt.ylabel("Cost function")
plt.show()

# Логарифмическая функция стоимости
fig6 = plt.figure(6)
fig6.canvas.set_window_title("Dynamics of cost function (LOGARIFMIC)")
neuron_3t = logn.create_logistic_neuron(n, cf.J_logarifmic)
costs_3 = []
neuron_3t.step_by_step_costs(X, y, batch_size, costs_3, learning_rate, eps, max_steps)
plt.plot(np.squeeze(costs_3))
plt.xlabel("Algorithm step number")
plt.ylabel("Cost function")
plt.show()