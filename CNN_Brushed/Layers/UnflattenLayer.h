#pragma once
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include "../Tensor.h"
#include "Layers.h"

class UnflattenLayer : public Layers {
public:
	int batchSize, inputSize;
	int outputChannels, outputHeight, outputWidth;

	UnflattenLayer(int outputChannels, int outputHeight, int outputWidth);

	std::unordered_map<std::string, int> initSizes(std::unordered_map<std::string, int>& sizes) override;

	Tensor forward(const Tensor& input) override;

	Tensor backward(const Tensor& dy) override;

	void gradientDescent(double alpha) override;

	void saveWeights(const std::string& filename) override;

	void loadWeights(const std::string& filename) override;

	//testing
	void addStuff(std::vector<double>& dO) override;
	//testing
};