#pragma once
#include <vector>
#include <Eigen/Dense>
#include <unordered_map>

#include "DenseLayer.h"
#include "ConvoLayer.h"
#include "FlattenLayer.h"
#include "MaxPoolLayer.h"


#include "ImageLoader.h"

#include "Loss.h"
#include <algorithm>

class NNModel {
private:

	int batchSize = -1;

	double datasetSize = 0;

	std::unordered_map<std::string, int> classNames;

	Tensor propagateInput(const Tensor& x);

	void propagateGradient(const Tensor& dy);

	void propagateSize(const std::unordered_map<std::string, int>& sizes);

	Eigen::MatrixXd calcCostGradient(Eigen::MatrixXd yHat, std::vector<double> y);

	Eigen::MatrixXd softmax(const Eigen::MatrixXd& x);

	Eigen::MatrixXd derivative_softmax_cross_entropy(const Eigen::MatrixXd& softmax_prob, const Eigen::VectorXi& labels);

	//void adamOptimizer(double alpha, double T, double e = 10e-7, double beta1 = 0.9, double beta2 = 0.999);

public:

	std::vector<std::shared_ptr<Layers>> layers;

	double modelAccuracy = 0;

	NNModel(const std::vector<std::shared_ptr<Layers>>& layersInput);

	double calcCost(Eigen::MatrixXd x, std::vector<double> y);

	double calcCost(std::vector < std::vector < Eigen::MatrixXd > > x, std::vector<std::string> yTrue);

	double calcBatchCost(const Eigen::MatrixXd& yHat, const Eigen::VectorXi& labels);

	// compilation for 1d inputs 
	void compile(int batchSize1, int inputSize);

	// compilation for 2d inputs (images)
	void compile(int batchSize1, int inputChannels, int inputHeight, int inputWidth);

	void fit(std::vector<std::vector<double>> input, std::vector<double> y, int epochs, double alpha, bool shuffle = false);

	Eigen::MatrixXd softmaxGradient(const Eigen::MatrixXd& yHat, const Eigen::VectorXi& labels);

	void train(std::vector<std::vector<Eigen::MatrixXd>> dataSet, std::vector<std::string> dataLabels);

	void fit(std::string path, int epochs, std::vector<std::string> classNamesS);

	// Use templates maybe?
	Eigen::MatrixXd predict(Eigen::MatrixXd x);

	Eigen::MatrixXd predict(std::vector < std::vector < Eigen::MatrixXd > > x);

	double calcAccuracy(std::vector<std::vector<double>> input, std::vector<double> y, double delimiter);

	void calcAccuracy(std::vector<std::vector<Eigen::MatrixXd>>& dataSet, std::vector<std::string>& dataLabels);

	double accuracy(std::string path);

	void loadWeights(const std::string& filename);

	void saveWeights(const std::string& filename);
};