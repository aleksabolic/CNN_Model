#pragma once
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include "Layers.h"

// I dont need to save the input ??
class MaxPoolLayer : public Layers{
public:
	int kernelSize, batchSize, stride;
	std::vector<std::vector<Eigen::MatrixXd>> layerOutput, outputGradients, gradGate; //(batch_size, numChannels, outputHeight, outputWidth)
	//std::vector<std::vector<Eigen::MatrixXd>> x; // (batch_size, channels, h, w)

	MaxPoolLayer(int kernelSize, int stride);

	std::unordered_map<std::string, int> initSizes(std::unordered_map<std::string, int>& sizes) override;

	Tensor forward(const Tensor& inputTensor) override;

	Tensor backward(const Tensor& dyTensor) override;

	void gradientDescent(double alpha) override;

	void saveWeights(const std::string& filename) override;

	void loadWeights(const std::string& filename) override;

	//testing
	void addStuff(std::vector<double>& dO) override;
	//testing
};