/*! \file  neuron.h
Общий класс нейрона + функции вычисления градиента, аналитически и численно.
При наследовании необходимо реализовать функцию стоимости, активационную функцию и производную активационной функции
(для аналитического расчета градиента).
*/
#pragma once
#include <Eigen/Dense>
#include <set>
#include <ctime>
#include <cstdlib>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Neuron {

public:
	/*
	Neuron
	w - вектор весов нейрона размерностью (n + 1, 1), где n - число признаков, w[0, 0] - смещение
	activation_function - активационная функция нейрона
	activation_function_derivative - производная активационной функции нейрона (для градиентного спуска)
	cost_function - функция стоимости
	*/
	Neuron(VectorXd w) {
		this->w = w;
	}
	
	// При наследовании от абстрактного класса нейрона необходимо реализовать следующие три метода:
	virtual VectorXd activation_function(VectorXd x) = 0;
	virtual VectorXd activation_function_derivative(VectorXd x) = 0;
	virtual double cost_function(Neuron &n, MatrixXd X, VectorXd y) = 0;

	// Внутри класса реализованы 2 функции стоимости: квадратичная и логарифмическая
	double J_quadratic(Neuron &n, MatrixXd X, VectorXd y);
	double J_logarifmic(Neuron &n, MatrixXd X, VectorXd y);

	/*
	forward_pass
	Активационная функция нейрона для единичного примера
	single_input - вектор размерностью (n + 1, 1), где n - число признаков
	*/
	double forward_pass(VectorXd single_input) {
		VectorXd sum = Eigen::VectorXd::Constant(1, 0.0);
		for (int i = 0; i != this->w.rows(); i++) {
			sum(0) += this->w(i) * single_input(i);
		}
		return activation_function(sum)(0,0);
	}
	/*
	summatory
	Сумматорная функция нейрона. Суммирует произведения весов и значений признаков для набора примеров.
	input_matrix - размерностью (m, n + 1), где m - число примеров, n - число признаков.
	Возвращает вектор сумм для набора примеров размерностью (m, 1).
	*/
	VectorXd summatory(MatrixXd input_matrix) {
		return input_matrix * this->w;
	}
	/*
	activation
	Активационная функция нейрона для набора примеров. Получает на вход результат работы сумматорной функции,
	вектор summatory_activation размерностью (m, 1), где m - число примеров.
	Возвращает вектор (m, 1) со значениями активации для всех примеров.
	*/
	VectorXd activation(VectorXd summatory_activation) {
		return activation_function(summatory_activation);
	}
	/*
	vectorized_forward_pass
	Векторизованная активационная функция для набора примеров.
	input_matrix - матрица размерностью (m, n + 1), где m - число примеров, n - число признаков.
	Возвращает вектор (m, 1) со значениями активации для всех примеров.
	*/
	VectorXd vectorized_forward_pass(MatrixXd input_matrix) {
		return activation(summatory(input_matrix));
	}
	/*
	GD
	Внешний цикл алгоритма градиентного спуска.
	X - матрица входных активаций (m, n + 1)
	y - вектор правильных ответов (m, 1)
	learning_rate - константа скорости обучения
	batch_size - размер обучающей выборки
	eps - критерий остановки - величина разности между значениями целевой функции для соседних итераций
	max_steps - критерий остановки №2 - максимально возможное число итераций
	Метод возвращает 1, если сработал первый критерий остановки (алгоритм сошелся), и 0,
	если сработал второй критерий (алгоритм не успел сойтись)
	*/
	int GD(MatrixXd X, VectorXd y, std::vector<double> &costs, int batch_size = 2, double learning_rate = 1.0, double eps = 1e-15, int max_steps = 200) {
		for (int step = 0; step < max_steps; step++) {
			std::set<int> numbers;
			srand((unsigned int)time(NULL));
			int m = X.rows();
			int n = X.cols();
			while (numbers.size() != batch_size) {
				int idx = rand() % m;
				numbers.insert(idx);
			}
			std::set<int>::iterator it; 
			MatrixXd batchX = Eigen::MatrixXd::Constant(batch_size, n, 1.0);
			VectorXd batchy = Eigen::VectorXd::Constant(batch_size, 0.0);
			int i = 0;
			for (it = numbers.begin(); it != numbers.end(); ++it) {
				//batchX.block(i, 0, 1, m) = X.block(*it, 0, 1, m);
				for (int j = 0; j < n; j++)
					batchX(i, j) = X(*it, j);
				batchy(i) = y(*it);
				i++;
			}
			double cost = cost_function(*this, X, y);
			costs.push_back(cost);
			if (update_mini_batch(batchX, batchy, learning_rate, eps))
				return 1;
		}
		return 0;
	}
	/*
	update_mini_batch
	Внутренняя часть GD - рассчитывает градиент и обновляет веса нейрона.
	X - матрица размера (batch_size, n + 1)
	y - вектор правильных ответов размера (batch_size, 1)
	learning_rate - константа скорости обучения
	eps - критерий остановки номер один: если разница между значением целевой функции 
	до и после обновления весов меньше eps - алгоритм останавливается.
	Возвращает 1 при срабатывании критерия, 0 в остальных случаях
	*/
	int update_mini_batch(MatrixXd X, VectorXd y, double learning_rate, double eps) {
		double J = cost_function(*this, X, y);
		VectorXd grad = compute_grad_analitically(*this, X, y);
		this->w -= learning_rate * grad;
		return (abs(J - cost_function(*this, X, y)) < eps);
	}

	/*
	Вычисляет вектор частных производных функции стоимости для каждого наблюдения
	*/
	VectorXd J_derivative(VectorXd y, VectorXd y_hat) {
		return (y_hat - y)/y.rows();
	}

	/*
	compute_grad_analitically
	Аналитическая производная функции стоимости.
    neuron - объект класса Neuron
    X - вертикальная матрица входов формы (m, n + 1), на которой считается сумма квадратов отклонений
    y - правильные ответы для примеров из матрицы X
    J_prime - функция, считающая производные целевой функции по ответам
    Возвращает вектор размера (m, 1)
	*/
	VectorXd compute_grad_analitically(Neuron &n, MatrixXd X, VectorXd y) {
		VectorXd z = n.summatory(X); // z - вектор результатов сумматорной функции нейрона на разных примерах
		VectorXd y_hat = n.activation(z); // значение активационной функции для всех примеров
		VectorXd dy_dyhat = J_derivative(y, y_hat); // производная функции стоимости
		VectorXd dyhat_dz = n.activation_function_derivative(z); // производная активационной функции
		MatrixXd dz_dw = X; // производная от аргумента активационной функции (x*w)' = x
		VectorXd mult = dy_dyhat *  dyhat_dz; 
		VectorXd grad = mult.transpose()*dz_dw; // вычисление сложной производной
		return grad.transpose(); // переход от строки к столбцу
	}
	/*
	compute_grad_numerically_2
	Численная производная функции стоимости
    neuron - объект класса Neuron с вертикальным вектором весов w,
    X - вертикальная матрица входов формы (n, m), на которой считается сумма квадратов отклонений,
    y - правильные ответы для тестовой выборки X,
    J - целевая функция, градиент которой мы хотим получить,
    eps - размер delta w (малого изменения весов).
    Возвращает вектор размера (m, 1)
	*/
	VectorXd compute_grad_numerically_2(Neuron &n, MatrixXd X, VectorXd y, double eps=1e-9) {
		VectorXd w_0 = n.w;
		VectorXd num_grad2 = Eigen::VectorXd::Constant(n.w.rows(), 0.0);
		for (int i = 0; i != n.w.rows(); i++) {
			double old_w = w_0(i);
			n.w(i) -= eps;
			double Jminus = n.cost_function(n, X, y);
			n.w(i) = old_w + eps;
			double Jplus = n.cost_function(n, X, y);
			num_grad2(i) = (Jplus - Jminus)/(2*eps);
			n.w(i) = old_w;
		}
		return num_grad2;
	}
	VectorXd w;
};