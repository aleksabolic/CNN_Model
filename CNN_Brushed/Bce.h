#pragma once
#include "Loss.h"
#include <vector>
#include <Eigen/Dense>
#include <iostream>

class BinaryCrossEntropy : public Loss {
public:
	BinaryCrossEntropy();
	double loss(double yHat, double yTrue) override;
	Eigen::MatrixXd gradient(Eigen::MatrixXd yHat, std::vector<double> y) override;
	double cost(Eigen::MatrixXd x, std::vector<double> y) override;

	// scce
	double cost(const Eigen::MatrixXd& yHat, const Eigen::VectorXi& labels) override;
	Eigen::MatrixXd gradient(const Eigen::MatrixXd& yHat, const Eigen::VectorXi& labels) override;

};