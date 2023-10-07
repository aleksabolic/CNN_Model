#pragma once
#include "Loss.h"
#include <vector>
#include <Eigen/Dense>
#include <iostream>

class BinaryCrossEntropy : public Loss {
public:
	BinaryCrossEntropy();
	double loss(double yHat, double yTrue) override;

	Eigen::MatrixXd gradient(const Eigen::MatrixXd& yHat,const std::vector<int>& y) override;
	double cost(const Eigen::MatrixXd& yHat, const std::vector<int>& y) override;
};