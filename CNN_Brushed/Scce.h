#pragma once
#include "Loss.h"
#include <iostream>

class SparseCategoricalCrossEntropy : public Loss {
public:
	SparseCategoricalCrossEntropy();
	double loss(double yHat, double yTrue) override;

	Eigen::MatrixXd gradient(const Eigen::MatrixXd& yHat, const std::vector<int>& y) override;
	double cost(const Eigen::MatrixXd& yHat, const std::vector<int>& y) override;
};