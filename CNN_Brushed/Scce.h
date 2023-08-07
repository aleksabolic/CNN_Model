#pragma once
#include "Loss.h"
#include <iostream>

class SparseCategoricalCrossEntropy : public Loss {
public:
	SparseCategoricalCrossEntropy();
	double loss(double yHat, double yTrue) override;
	double cost(const Eigen::MatrixXd& yHat, const Eigen::VectorXi& labels) override;
	Eigen::MatrixXd gradient(const Eigen::MatrixXd& yHat, const Eigen::VectorXi& labels) override;

	// bce
	Eigen::MatrixXd gradient(Eigen::MatrixXd yHat, std::vector<double> y) override;
	double cost(Eigen::MatrixXd yHat, std::vector<double> y) override;
};