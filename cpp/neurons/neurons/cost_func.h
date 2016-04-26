/*! \file  cost_func.h
Функции стоимости (целевые функции) для решения задач регрессии и классификации.
*/
#pragma once
#include <iostream>
#include <Eigen/Dense>
#include "neuron.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;

double Neuron::J_quadratic(Neuron &n, MatrixXd X, VectorXd y) {
	VectorXd prediction = n.vectorized_forward_pass(X);
	double mult = ((prediction - y).transpose() * (prediction - y));
	return 0.5 * mult / X.rows();
}

double Neuron::J_logarifmic(Neuron &n, MatrixXd X, VectorXd y) {
	VectorXd prediction = n.vectorized_forward_pass(X);
	VectorXd tmp = Eigen::VectorXd::Constant(prediction.rows(), 1.0);
	for (int i = 0; i != prediction.rows(); i++)
		tmp(i) = log(prediction(i));
	double mult1 = (y.transpose()  * tmp)(0,0);
	VectorXd npd1 = -y.transpose() + Eigen::VectorXd::Constant(prediction.rows(), 1.0); 
	VectorXd npd2 = -prediction + Eigen::VectorXd::Constant(prediction.rows(), 1.0); 
	for (int i = 0; i != npd2.rows(); i++)
		npd2(i) = log(npd2(i));
	double mult2 = (npd1 * npd2)(0,0);
	return -(mult1 + mult2) / X.rows();
}