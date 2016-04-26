/*! \file  logistic_neuron.h
Класс логистического нейрона.
*/
#pragma once
#include "neuron.h"
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class LogisticNeuron : public Neuron {
public:
	LogisticNeuron(VectorXd w, int cost_f = 1): Neuron(w) {
		this->cost_f = cost_f;
	}
	VectorXd activation_function(VectorXd x) {
		VectorXd res = -x;
		for (int i = 0; i != res.rows(); i++) {
			res(i) = 1/ (exp(res(i)) + 1);
		}
		return res;
	}
	VectorXd activation_function_derivative(VectorXd x) {
		VectorXd diff = -activation_function(x) + Eigen::VectorXd::Constant(x.rows(), 1.0);
		return activation_function(x) * diff;
	}
	double cost_function(Neuron &n, MatrixXd X, VectorXd y) {
		if (cost_f == 1)
			return J_logarifmic(n, X, y);
		else
			return J_quadratic(n, X, y);
	}

	int cost_f;
};

/*
create_logistic_neuron
Создаем логистический нейрон со случайными весами и единичным смещением. 
type - тип функции стоимости (1 - логарифмическая, 2 - квадратичная)
*/
LogisticNeuron create_logistic_neuron(int n, int type = 1) {
	VectorXd w = VectorXd::Random(n + 1);
	w(0) = 1;
	return LogisticNeuron(w, type);
}