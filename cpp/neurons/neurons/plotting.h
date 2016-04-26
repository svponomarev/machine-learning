/*! \file  logistic_neuron.h
Набор процедур для визуализации результатов работы алгоритмов регрессии и классификации.
*/
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <discpp.h>
#include "perceptron.h"
#include "neuron.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

double plot_regression(int id, char *title, char *xlabel, char *ylabel, MatrixXd X, VectorXd y, Neuron &n);
double plot_classification(int id, char *title, char *xlabel, char *ylabel, MatrixXd X, VectorXd y);
void plot_decision_line(Perceptron &p, double scale, char *color);
void plot_decision_line(Neuron &n, double scale, char *color);
void plot_costf_dynamics(int id, char *title, char *xlabel, char *ylabel, std::vector<double> costs);