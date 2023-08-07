#pragma once
#include <math.h>
#include <Eigen/Dense>

class Loss {
public:
	virtual ~Loss() {}

	virtual double loss(double yHat, double y) = 0;

	// bce
	virtual Eigen::MatrixXd gradient(Eigen::MatrixXd yHat, std::vector<double> y) = 0;
	virtual double cost(Eigen::MatrixXd yHat, std::vector<double> y) = 0;

	// scce
	virtual double cost(const Eigen::MatrixXd& yHat, const Eigen::VectorXi& labels) = 0;
	virtual Eigen::MatrixXd gradient(const Eigen::MatrixXd& yHat, const Eigen::VectorXi& labels) = 0;
};

