/*! \file  perceptron.h
Класс перцептрона со встроенной функцией для визуализации динамики обучения (график количества ошибок от числа итераций).
*/
#pragma once
#include <Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Perceptron {

public:
	/*
	Perceptron
	Инициализация перцептрона
	w - вектор весов размерностью (n + 1, 1), n - число признаков
	*/
	Perceptron(VectorXd w) {
		this->w = w;
	}
	/*
	forward_pass
	Расчет ответа перцептрона при предъявлении одного примера
    single_input - вектор примера размерностью (n + 1, 1)
    Метод возвращает True/False 
	*/
	int forward_pass(VectorXd single_input) {
		double result = 0;
		for (int i = 0; i != this->w.rows(); i++) {
			result += (this->w(i) * single_input(i));
		}
		return (result > 0);
	}
	/*
	vectorized_forward_pass
	Расчет вектора ответов перцептрона при предъявлении набора примеров
	input_matrix - матрица размерностью (m, n + 1), где m - число наблюдений,
	n - число признаков
	Метод возвращает вектор ответов перцептрона (True/False) размерностью (m, 1)
	*/
	MatrixXd vectorized_forward_pass(MatrixXd input_matrix) {
		MatrixXd mult = input_matrix * this->w;
		MatrixXd result = Eigen::MatrixXd::Constant(mult.rows(), 1,  0.0);
		for (int i = 0; i != mult.rows(); i++) {
			if (mult(i) > 0)
				result(i) = 1;
		}
		return result;
	}
	/*
	train_on_single_example
	Принимает вектор активации входов example (n + 1, 1)
	и ожидаемый ответ y (0 / 1). Обновляет значения весов в случае ошибки.
	Возвращает размер ошибки (0 или 1).
	*/
	int train_on_single_example(MatrixXd example, int y) {
		MatrixXd response = vectorized_forward_pass(example);
		int r = (int)response(0, 0);
		if (r != y) {
			this->w += (y - r) * example;
			return 1;
		}
		else
			return 0;
	}

	double count_errors(MatrixXd input_matrix, VectorXd y) {
		VectorXd prediction = vectorized_forward_pass(input_matrix);
		return (prediction - y).cwiseAbs().sum();
	}
	/*
	train_until_convergence
	Обучение перцептрона на наборе примеров.
	input_matrix - матрица примеров (m, n + 1), где
	m - число примеров, n - число признаков
	y - вектор с правильными ответами (True/False) размерностью (m, 1)
	max_steps - максимальное количество итераций алгоритма
	*/
	int train_until_convergence(MatrixXd input_matrix, VectorXd y, std::vector<double> &err, int max_steps = 10000) {
		int i = 0;
		int errors = 1;
		err.push_back(count_errors(input_matrix, y));
		while ((errors != 0) && (i < max_steps)) {
			i++;
			errors = 0;
			for (int j = 0; j != input_matrix.rows(); j++) {
				MatrixXd example = input_matrix.block(j, 0, 1, input_matrix.cols());
				int answer = (int)y(j);
				int error = train_on_single_example(example, answer);
				errors += error;
			}
			err.push_back(count_errors(input_matrix, y));
		}
		if (errors == 0)
			return 1;
		else
			return 0;
	}

	VectorXd w;
};

/*
create_perceptron
Создаем перцептрон со случайными весами и единичным смещением.
*/
Perceptron create_perceptron(int n) {
	VectorXd w = VectorXd::Random(n + 1);
	w(0) = 1;
	return Perceptron(w);
}