#include <iostream>
#include <iomanip>
#include <fstream>
#include <Eigen/Dense>
#include <discpp.h>
#include "CSVRow.h"
#include "perceptron.h"
#include "cost_func.h"
#include "linear_neuron.h"
#include "logistic_neuron.h"
#include "plotting.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

Dislin g; // объект, осуществляющий все функции библиотеки построения графиков DISLIN

/*
	Набор процедур для визуализации результатов работы алгоритмов регрессии и классификации.
*/
double plot_regression(int id, char *title, char *xlabel, char *ylabel, MatrixXd X, VectorXd y, Neuron &n);
double plot_classification(int id, char *title, char *xlabel, char *ylabel, MatrixXd X, VectorXd y);
void plot_decision_line(Perceptron &p, double scale, char *color);
void plot_decision_line(Neuron &n, double scale, char *color);
void plot_costf_dynamics(int id, char *title, char *xlabel, char *ylabel, std::vector<double> costs);

/**
	Загрузка из CSV-файла данных в матрицу признаков X размерностью (m, n + 1) и вектор правильных ответов (m, 1)
**/
void load_data(char *filename, MatrixXd &X, VectorXd &y) {
	std::ifstream file(filename);
	CSVRow row;
	std::vector<CSVRow> rows;
	while (file >> row) {
		rows.push_back(row);
	}
	int m = rows.size(); // m - число наблюдений
	int n = rows[0].size() - 1; // n - число признаков, 1 колонка отводится на единичное смещение
	y = Eigen::VectorXd::Constant(m, 0.0);
	X =  Eigen::MatrixXd::Constant(m, n + 1, 1.0);
	for (int i = 0; i != rows.size(); i++) {
		y(i) = stod(rows[i][n]);
		for (int j = 0; j < n; j++) {
			X(i, 1 + j) = stod(rows[i][j]);
		}
	}
}

/**
	Форматированный вывод для классификации по 2 признакам.
**/
void formatted_output(Neuron &n, MatrixXd X, VectorXd y, int batch_size, std::vector<double> &costs, double learning_rate, double eps, int max_steps) {

	std::cout << "Equation BEFORE learning: " << n.w(2) << "*x1 + " << n.w(1) << "*x2 = " << -n.w(0)  << std::endl;
	std::cout << "Cost function BEFORE learning: " << n.cost_function(n, X, y) << std::endl;
	std::cout << "Is neuron converge = " << n.GD(X, y, costs, batch_size, learning_rate, eps, max_steps) << std::endl;
	std::cout << "Equation AFTER learning: " << n.w(2) << "*x1 + " << n.w(1) << "*x2 = " << -n.w(0) << std::endl;
	std::cout << "Cost function AFTER learning: " << n.cost_function(n, X, y) << std::endl;
}

int main()
{
	char *data_r = "..\\..\\..\\data\\regression_data.csv";
	MatrixXd X;	VectorXd y;
	std::vector<double> errors;
	std::vector<double> costs_1, costs_2, costs_3;


	// Инициализация DISLIN
	g.metafl ("xwin"); 
	g.scrmod ("revers"); // обратные цвета (по дефолту фон черный, а не белый)
	g.disini ();
	g.pagera ();
	g.complx ();
	g.errmod("ALL", "OFF"); // убрать вывод всех предупреждений

	// Линейная регрессия
	load_data(data_r, X, y);
	std::cout << "REGRESSION:" << std::endl;
	std::cout << "Matrix X: [" << X.rows() << "x" << X.cols() << "]" << std::endl;
	std::cout << "Vector y: [" << y.rows() << "x" << y.cols() << "]" << std::endl << std::endl;
	int m = X.rows();
	int n = X.cols() - 1;
	// Создание линейного нейрона
	LinearNeuron neuron_1 = create_linear_neuron(n);
	int batch_size = 40; // batch_size - размер одной "порции" примеров для итерации градиентного спуска
	double learning_rate = 0.01; // learning_rate - коэффициент скорости обучения
	double eps = 1e-5; // eps - минимальная разность в функции стоимости, при которой будет продолжена работа алгоритма
	int max_steps = 10000; // max_steps - максимальное число итераций градиентного спуска

	std::cout << "LINEAR NEURON:" << std::endl;
	std::cout << "Equation BEFORE learning: y = " << neuron_1.w(0) << "*x + " << neuron_1.w(1) << std::endl;
	std::cout << "Cost function BEFORE learning: " << neuron_1.cost_function(neuron_1, X, y) << std::endl;
	std::cout << "Is perceptron converge = " << neuron_1.GD(X, y, costs_1, batch_size, learning_rate, eps, max_steps) << std::endl;
	std::cout << "Equation AFTER learning: y = " << neuron_1.w(0) << "*x + " << neuron_1.w(1) << std::endl;
	std::cout << "Cost function AFTER learning: " << neuron_1.cost_function(neuron_1, X, y) << std::endl << std::endl;

	plot_regression(1, "Profit of city estimation", "Population of city in 10,000s", "Profit in $10,000s", X, y, neuron_1);

	// Бинарная классификация по 2 признакам
	char *data_c = "..\\..\\..\\data\\classification_data.csv";
	load_data(data_c, X, y);
	std::cout << "CLASSIFICATION:" << std::endl;
	std::cout << "Matrix X: [" << X.rows() << "x" << X.cols() << "]" << std::endl;
	std::cout << "Vector y: [" << y.rows() << "x" << y.cols() << "]" << std::endl << std::endl;
	m = X.rows();
	n = X.cols() - 1;

	// Построение данных для классификации - scatter plot. В scale находится максимальное значение по x, для построения разделяющих прямых
	double scale = plot_classification(2, "Apple/Pears classification", "Yellowness", "Symmetry", X, y);
	
	// Создание перцептрона
	Perceptron p = create_perceptron(n);
	std::cout << "PERCEPTRON:" << std::endl;
	std::cout << std::fixed << std::setprecision(6);
	std::cout << "Equation BEFORE learning: " << p.w(2) << "*x1 + " << p.w(1) << "*x2 = " << -p.w(0)  << std::endl;
	std::cout << "Is perceptron converge = " << p.train_until_convergence(X, y, errors) << std::endl;
	std::cout << "Equation AFTER learning: " << p.w(2) << "*x1 + " << p.w(1) << "*x2 = " << -p.w(0) << std::endl;
	std::cout << "Num. of errors BEFORE learning: " << errors[0] << std::endl;
	std::cout << "Num. of errors AFTER learning: " << errors[errors.size() - 1] << std::endl << std::endl;

	plot_decision_line(p, scale, "white");

	// Создание логистических нейронов
	LogisticNeuron neuron_2 = create_logistic_neuron(n); // cost_f = 1 - логистическая функция стоимости
	LogisticNeuron neuron_3 = create_logistic_neuron(n, 2); // cost_f = 2 - квадратичная функция стоимости
	
	batch_size = 40; // batch_size - размер одной "порции" примеров для итерации градиентного спуска
	learning_rate = 5.0; // learning_rate - коэффициент скорости обучения
	eps = 1e-12; // eps - минимальная разность в функции стоимости, при которой будет продолжена работа алгоритма
	max_steps = 10000; // max_steps - максимальное число итераций градиентного спуска

	std::cout << "LOGISTIC NEURON (LOGARIFMIC COST FUNCTION):" << std::endl;
	formatted_output(neuron_2, X, y,  batch_size, costs_2, learning_rate, eps, max_steps);
	std::cout << std::endl << "LOGISTIC NEURON (QUADRATIC COST FUNCTION):" << std::endl;
	formatted_output(neuron_3, X, y,  batch_size, costs_3, learning_rate, eps, max_steps);

	plot_decision_line(neuron_2, scale, "green");
	plot_decision_line(neuron_3, scale, "magenta");

	// Вывод легенды для графика классификации
	char cbuf[256];
	g.legini (cbuf, 3, 25);
	g.color("white");
	g.legtit ("");
	g.leglin (cbuf, "Perceptron", 1);
	g.leglin (cbuf, "Logistic neuron (quad)", 2);
	g.leglin (cbuf, "Logistic neuron (log)", 3);
	g.legend (cbuf, 3);
	g.endgrf(); // Заканчиваем рисовать текущий график

	// Вывод графиков для оценки изменения функции стоимости на каждой итерации
	//plot_costf_dynamics(g, 3, "Dynamics of cost function (LINEAR)", "Algorithm step number", "Cost function", costs_1);
	plot_costf_dynamics(3, "Dynamics of cost function (QUADRATIC)", "Algorithm step number", "Cost function", costs_2);
	plot_costf_dynamics(4, "Dynamics of cost function (LOGARIFMIC)", "Algorithm step number", "Cost function", costs_3);
	plot_costf_dynamics(5, "Dynamics of learning (PERCEPTRON)", "Algorithm step number", "Number of errors", errors);

	g.disfin (); // закрываем DISLIN

	return 0;
}


double plot_regression(int id, char *title, char *xlabel, char *ylabel, MatrixXd X, VectorXd y, Neuron &n) {
	int ic;
	char cbuf[256];
	double *xray = new double[X.rows()];
	double *y1ray = new double[X.rows()];

	double maxx = 0;
	double maxy = 0;
	for (int i = 0; i < X.rows(); i++)
	{   xray[i] = X(i, 1);
	y1ray[i] = y(i);
	if (xray[i] > maxx)
		maxx = xray[i];
	if (y1ray[i] > maxy)
		maxy = y1ray[i];
	}
	int k = (int)maxx;
	double x2ray[100];
	double y2ray[100];
	for (int i = 0; i < k; i++)
	{
		x2ray[i] = i;
		y2ray[i] = n.w(1) *i + n.w(0);
	}

	g.opnwin(id);

	g.axspos (450, 1800);
	g.axslen (2200, 1200);

	g.name   (xlabel, "x");
	g.name   (ylabel, "y");

	g.labdig (-1, "x");
	g.ticks  (9, "x");
	g.ticks  (10, "y");

	g.titlin (title, 3);

	ic=g.intrgb (0.95,0.95,0.95);
	g.axsbgd (ic);

	g.graf   (0.0, maxx, 0.0, maxx/5, 0, maxy, 0,  maxy/5);
	g.setrgb (0.7, 0.7, 0.7);
	g.grid   (1, 1);

	g.color  ("fore");
	g.height (50);
	g.title  ();

	g.color  ("blue");

	for  (int i = 0; i < X.rows(); i++)
		g.rlcirc ( xray[i], y1ray[i], 0.1 );

	g.color  ("red");
	g.curve  (x2ray, y2ray, k);

	g.color  ("white");
	g.legini (cbuf, 1, 20);
	g.legtit ("");
	g.leglin (cbuf, "Linear Regression", 1);
	g.legend (cbuf, 3);

	g.endgrf();

	delete xray;
	delete y1ray;

	return k;
}

double plot_classification(int id, char *title, char *xlabel, char *ylabel, MatrixXd X, VectorXd y) {

	int ic;
	double *x1ray = new double[X.rows()];
	double *x2ray = new double[X.rows()];

	double maxx = 0;
	double maxy = 0;
	for (int i = 0; i < X.rows(); i++)
	{   x1ray[i] = X(i, 1);
		x2ray[i] = X(i, 2);
	if (x1ray[i] > maxx)
		maxx = x1ray[i];
	if (x2ray[i] > maxy)
		maxy = x2ray[i];
	}

	g.opnwin(id);

	g.axspos (450, 1800);
	g.axslen (2200, 1200);

	g.name   (xlabel, "x");
	g.name   (ylabel, "y");

	g.labdig (-1, "x");
	g.ticks  (9, "x");
	g.ticks  (10, "y");

	g.titlin(title, 3);
	ic=g.intrgb (0.95,0.95,0.95);
	g.axsbgd (ic);

	g.graf   (0.0, maxx, 0.0, maxx/5, 0, maxy, 0,  maxy/5);
	g.setrgb (0.7, 0.7, 0.7);
	g.grid   (1, 1);

	g.color  ("fore");
	g.height (50);
	g.title  ();

	g.color  ("blue");
	for  (int i = 0; i < X.rows(); i++)
		if (y(i) == 0)
			g.rlcirc ( x1ray[i], x2ray[i], 0.01 );

	g.color  ("red");
	for  (int i = 0; i < X.rows(); i++)
		if (y(i) == 1)
			g.rlcirc ( x1ray[i], x2ray[i], 0.01 );

	delete x1ray;
	delete x2ray;

	return maxx;
}

void plot_decision_line(Perceptron &p, double scale, char *color) {
	double xray[100];
	double yray[100];
	double steps = scale/100;
	int index = 0;
	for (double i = 0; i < scale; i+=steps)
	{
		xray[index] = i;
		yray[index] = (-1/p.w(2))*(p.w(1) * i + p.w(0));
		index++;
	}
	g.thkcrv(1);
	g.color(color);
	g.curve  (xray, yray, index);
}

void plot_decision_line(Neuron &n, double scale, char *color) {
	double xray[100];
	double yray[100];
	double steps = scale/100;
	int index = 0;
	for (double i = 0; i < scale; i+=steps)
	{
		xray[index] = i;
		yray[index] = (-1/n.w(2))*(n.w(1) * i + n.w(0));
		index++;
	}
	g.thkcrv(1);
	g.color(color);
	g.curve  (xray, yray, index);
}

void plot_costf_dynamics(int id, char *title, char *xlabel, char *ylabel, std::vector<double> costs) {
	int ic = 0;
	double *xray = new double[costs.size()];
	double *yray = new double[costs.size()];
	double maxy = 0;
	for (unsigned int i = 0; i < costs.size(); i++) {
		xray[i] = i;
		yray[i] = costs[i];
		if (costs[i] > maxy)
			maxy = costs[i];
	}
	g.opnwin(id);

	g.axspos (450, 1800);
	g.axslen (2200, 1200);

	g.name   (xlabel, "x");
	g.name   (ylabel, "y");

	g.labdig (-1, "x");
	g.ticks  (9, "x");
	g.ticks  (10, "y");

	g.titlin(title, 3);
	ic=g.intrgb (0.95,0.95,0.95);
	g.axsbgd (ic);

	g.graf   (0.0, costs.size(), 0.0, costs.size()/5, 0, maxy, 0,  maxy/2);
	g.setrgb (0.7, 0.7, 0.7);
	g.grid   (1, 1);

	g.color  ("fore");
	g.height (50);
	g.title  ();

	g.color  ("blue");
	g.curve  (xray, yray, costs.size());

	delete xray;
	delete yray;
	g.endgrf();

	g.color  ("white");
}