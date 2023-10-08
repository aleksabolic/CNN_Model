#include "Scce.h"

SparseCategoricalCrossEntropy::SparseCategoricalCrossEntropy(){}

double SparseCategoricalCrossEntropy::loss(double yHat, double y) {
	return -log(yHat + 1e-7);
}

// This function expects yHat to be softmaxed
double SparseCategoricalCrossEntropy::cost(const Eigen::MatrixXd& yHat, const std::vector<int>& labels) {
	double cost = 0.0;

	double epsilon = 1e-10;

	//clip yHat to prevent log(0) errors
	Eigen::MatrixXd yHatClipped = yHat.cwiseMax(epsilon).cwiseMin(1.0 - epsilon);

	for (int z = 0; z < labels.size(); z++) {
		int yTrueIndex = labels[z];
		cost += -std::log(yHatClipped(z, yTrueIndex));
	}
	return cost / labels.size();
}

Eigen::MatrixXd SparseCategoricalCrossEntropy::gradient(const Eigen::MatrixXd& yHat, const std::vector<int>& labels) {

	Eigen::MatrixXd dy = Eigen::MatrixXd::Zero(yHat.rows(), yHat.cols());

	for (int z = 0; z < dy.rows(); z++) {
		int yTrueIndex = labels[z];
		dy(z, yTrueIndex) = 1.0;  // Construct one-hot encoded ground truth
	}

	return yHat - dy;  // Gradient is (yHat - yTrue)


	/*Eigen::MatrixXd dy = yHat;
	for (int z = 0; z < dy.rows(); z++) {
		int yTrueIndex = labels[z];
		dy(z, yTrueIndex) -= 1.0;
	}
	dy /= yHat.rows();
	return dy;*/
}
