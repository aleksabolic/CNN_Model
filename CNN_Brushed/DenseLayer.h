#pragma once
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <iostream>

#include "Tensor.h"
#include "Layers.h"

#include "Sigmoid.h"

class DenseLayer : public Layers{
private:
	int numNodes;
	int inputSize;
	int batchSize;
	std::string activation;
public:
	Eigen::MatrixXd x, w, WGradients, layerOutput, outputGradients, nodeGrads; //w is transposed by default
	Eigen::RowVectorXd b, BGradients;
	std::vector<Eigen::MatrixXd> softmaxNodeGrads;

	//Constructor
	DenseLayer(int numNodes, const std::string& activation);

	std::unordered_map<std::string, int> initSizes(std::unordered_map<std::string, int> sizes) override;

	void uploadWeightsBias(std::vector<std::vector<double>> wUpload, std::vector<double> bUpload);

	Tensor forward(Tensor inputTensor) override;

	Tensor backward(Tensor dyTensor) override;

	void gradientDescent(double alpha) override;

	void saveWeights(const std::string& filename) override;

	void loadWeights(const std::string& filename) override;
};