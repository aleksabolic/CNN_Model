#pragma once
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <iostream>

#include "../Tensor.h"
#include "Layers.h"

#include "../Sigmoid.h"

class DenseLayer : public Layers{
private:
	int numNodes;
	int inputSize;
	int batchSize;
	int t;
	std::string activation;
	bool regularization;
public:
	//removed wgradients and bgradients for testing
	Eigen::MatrixXd x, layerOutput, outputGradients, nodeGrads, vdw, sdw; //w is transposed by default
	Eigen::VectorXd vdb, sdb;
	std::vector<Eigen::MatrixXd> softmaxNodeGrads;

	//Constructor
	DenseLayer(int numNodes, const std::string& activation, bool regularizaton = false);

	std::unordered_map<std::string, int> initSizes(std::unordered_map<std::string, int>& sizes) override;

	//<---------change with initCustomWeights()---------->
	void uploadWeightsBias(std::vector<std::vector<double>> wUpload, std::vector<double> bUpload);

	Tensor forward(const Tensor& inputTensor) override;

	Tensor backward(const Tensor& dyTensor) override;

	void gradientDescent(double alpha) override;

	void saveWeights(const std::string& filename) override;

	void loadWeights(const std::string& filename) override;

	//testing
	void addStuff(std::vector<double>& dO) override;
	//testing
};