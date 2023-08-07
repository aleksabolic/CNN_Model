#pragma once
#include <vector>
#include <Eigen/Dense>
#include <unordered_map>

#include "./Layers/DenseLayer.h"
#include "./Layers/ConvoLayer.h"
#include "./Layers/FlattenLayer.h"
#include "./Layers/MaxPoolLayer.h"


#include "ImageLoader.h"

#include "Loss.h"
#include <algorithm>

class NNModel {
private:

	int batchSize = -1;

	double datasetSize = 0;

	Loss* loss_ptr;

	std::unordered_map<std::string, int> classNames;

	Tensor propagateInput(const Tensor& x);

	void propagateGradient(const Tensor& dy);

	void propagateSize(const std::unordered_map<std::string, int>& sizes);

	Eigen::MatrixXd softmax(Eigen::MatrixXd x);

public:

	std::vector<std::shared_ptr<Layers>> layers;

	double modelAccuracy = 0;

	NNModel(const std::vector<std::shared_ptr<Layers>>& layersInput);

	// compilation for 1d inputs 
	void compile(int batchSize1, int inputSize, Loss* loss_pointer);

	// compilation for 2d inputs (images)
	void compile(int batchSize1, int inputChannels, int inputHeight, int inputWidth, Loss* loss_pointer);

	void fit(std::vector<std::vector<double>> input, std::vector<double> y, int epochs, double alpha, bool shuffle = false);

	void train(std::vector<std::vector<Eigen::MatrixXd>>& dataSet, std::vector<std::string>& dataLabels);

	void fit(std::string path, int epochs, std::vector<std::string> classNamesS);

	// Use templates maybe?
	Eigen::MatrixXd predict(Eigen::MatrixXd x);

	Eigen::MatrixXd predict(std::vector < std::vector < Eigen::MatrixXd > > x);

	double calcAccuracy(std::vector<std::vector<double>> input, std::vector<double> y, double delimiter);

	void calcAccuracy(std::vector<std::vector<Eigen::MatrixXd>>& dataSet, std::vector<std::string>& dataLabels);

	double accuracy(std::string path, std::vector<std::string> classNamesS);

	void loadWeights(const std::string& filename);

	void saveWeights(const std::string& filename);

	//testing 
	void checkGrad(std::vector<std::vector<Eigen::MatrixXd>>& dataSet, std::vector<std::string>& dataLabels);

	void gradientChecking(std::string path, std::vector<std::string> classNamesS);
	void gradientChecking(std::vector<std::vector<double>> x, std::vector<double> y);
	//testing
};