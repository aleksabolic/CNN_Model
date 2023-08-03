#pragma once
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <algorithm>
#include <utility>

#include "Layers.h"
#include "../Tensor.h"

class ConvoLayer : public Layers{
private:
	int numFilters, kernelSize, padding, batchSize;
	std::pair<int, int> strides;
	std::string activation;
	bool regularization;

	std::vector<std::vector<Eigen::MatrixXd>> W, WGradients; // (numFilters, channels, h, w)
	std::vector<std::vector<Eigen::MatrixXd>> layerOutput, outputGradients, nodeGrads; //(batch_size, numFilters, outputHeight, outputWidth)
	Eigen::VectorXd b, BGradients; //(numFilters)
	std::vector<std::vector<Eigen::MatrixXd>> x; // (batch_size, channels, h, w)
public:

	ConvoLayer(int numFilters, int kernelSize, std::pair<int, int> strides, int padding, std::string activation, bool regularization = false);

	std::unordered_map<std::string, int> initSizes(std::unordered_map<std::string, int>& sizes) override;

	Tensor forward(const Tensor& inputTensor) override;

	Tensor backward(const Tensor& dyTensor) override;

	void gradientDescent(double alpha) override;

	void saveWeights(const std::string& filename) override;

	void loadWeights(const std::string& filename) override;
};