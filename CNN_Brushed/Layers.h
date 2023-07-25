#pragma once
#include "Tensor.h"
class Layers {
public:
	virtual ~Layers() {}
	virtual Tensor forward(Tensor inputTensor) = 0;
	virtual Tensor backward(Tensor dyTensor) = 0;
	virtual std::unordered_map<std::string, int> initSizes(std::unordered_map<std::string, int> sizes) = 0;
};