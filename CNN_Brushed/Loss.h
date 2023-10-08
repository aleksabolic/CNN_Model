#pragma once
#include <math.h>
#include <Eigen/Dense>

class Loss {
public:
	virtual ~Loss() {}

	virtual double loss(double yHat, double y) = 0;

	virtual Eigen::MatrixXd gradient(const Eigen::MatrixXd& yHat, const std::vector<int>& y) = 0;
	virtual double cost(const Eigen::MatrixXd& yHat, const std::vector<int>& y) = 0;

};

