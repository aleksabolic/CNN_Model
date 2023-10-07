#include "Bce.h"

BinaryCrossEntropy::BinaryCrossEntropy() {}

double BinaryCrossEntropy::loss(double yHat, double y) {
	double eps = 1e-7;
	yHat = std::max(std::min(yHat, 1 - eps), eps);
	return -y * log(yHat) - (1.0 - y) * log(1.0 - yHat);
}

Eigen::MatrixXd BinaryCrossEntropy::gradient(const Eigen::MatrixXd& yHat, const std::vector<int>& y) {

	double eps = 1e-7;
	Eigen::MatrixXd gradients = Eigen::MatrixXd(yHat.rows(), 1);
	for (int i = 0; i < y.size(); i++) {
		double x = std::max(std::min(yHat(i, 0), 1 - eps), eps);
		gradients(i, 0) = -y[i] / x + (1 - y[i]) / (1.0 - x);
	}
	return gradients;
}

double BinaryCrossEntropy::cost(const Eigen::MatrixXd& yHat,const std::vector<int>& y) {
	double cost = 0.0;

	for (int i = 0; i < y.size(); i++) {
		cost += loss(yHat(i, 0), y[i]);
	}
	return cost / y.size();
}

