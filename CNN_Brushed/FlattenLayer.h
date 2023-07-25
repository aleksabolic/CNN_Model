#pragma once
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include "Tensor.h"
#include "Layers.h"

class FlattenLayer : public Layers{
public:
	int batchSize, inputChannels, inputHeight, inputWidth;

	std::unordered_map<std::string, int> initSizes(std::unordered_map<std::string, int> sizes) override;

	Tensor forward(Tensor input) override;

	Tensor backward(Tensor dy) override;
};