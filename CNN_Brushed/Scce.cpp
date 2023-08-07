#include "Scce.h"

SparseCategoricalCrossEntropy::SparseCategoricalCrossEntropy(){}

double SparseCategoricalCrossEntropy::loss(double yHat, double y) {
	return -log(yHat + 1e-7);
}

// This function expects yHat to be softmaxed
double SparseCategoricalCrossEntropy::cost(const Eigen::MatrixXd& yHat, const Eigen::VectorXi& labels) {
	double cost = 0.0;

	double epsilon = 1e-7;

	for (int z = 0; z < labels.size(); z++) {
		int yTrueIndex = labels(z);
		cost += -std::log(yHat(z, yTrueIndex) + epsilon);
	}
	return cost / labels.size();
}

Eigen::MatrixXd SparseCategoricalCrossEntropy::gradient(const Eigen::MatrixXd& yHat, const Eigen::VectorXi& labels) {
	Eigen::MatrixXd dy = Eigen::MatrixXd::Zero(yHat.rows(), yHat.cols());

	for (int z = 0; z < dy.rows(); z++) {
		int yTrueIndex = labels[z];
		dy(z, yTrueIndex) = 1.0;  // Construct one-hot encoded ground truth
	}

	return yHat - dy;  // Gradient is (yHat - yTrue)
}

// bce
Eigen::MatrixXd SparseCategoricalCrossEntropy::gradient(Eigen::MatrixXd yHat, std::vector<double> y) {
	return Eigen::MatrixXd::Zero(1, 1);
}
double SparseCategoricalCrossEntropy::cost(Eigen::MatrixXd yHat, std::vector<double> y) {
	return -INFINITY;
}