/*! \file  linear_neuron.h
Класс линейного нейрона.
*/
#pragma once
#include "neuron.h"
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class LinearNeuron : public Neuron {
public:
	LinearNeuron(VectorXd w): Neuron(w) {
	}

	VectorXd activation_function(VectorXd x) {
		return x;
	}
	VectorXd activation_function_derivative(VectorXd x) {
		return Eigen::VectorXd::Constant(x.rows(), 1.0);
	}
	double cost_function(Neuron &n, MatrixXd X, VectorXd y) {
		return J_quadratic(n, X, y);
	}
};
/*
create_linear_neuron
Создаем линейный нейрон со случайными весами и единичным смещением. 
*/
LinearNeuron create_linear_neuron(int n) {
	VectorXd w = VectorXd::Random(n + 1);
	w(0) = 1;
	return LinearNeuron(w);
}