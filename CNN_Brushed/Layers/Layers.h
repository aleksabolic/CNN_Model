#pragma once
#include "../Tensor.h"
class Layers {
public:
	bool trainable;

	//testing
	Eigen::MatrixXd W;
	Eigen::VectorXd b;
	//testing

	virtual ~Layers() {}
	virtual Tensor forward(const Tensor& inputTensor) = 0;
	virtual Tensor backward(const Tensor& dyTensor) = 0;
	virtual std::unordered_map<std::string, int> initSizes(std::unordered_map<std::string, int>& sizes) = 0;
	virtual void gradientDescent(double alpha) = 0;
	virtual void saveWeights(const std::string& filename) = 0;
	virtual void loadWeights(const std::string& filename) = 0;

	//testing
	virtual void addStuff(std::vector<double>& dO) = 0;
	//testing
};