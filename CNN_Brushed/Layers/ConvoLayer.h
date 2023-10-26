#pragma once
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <algorithm>
#include <utility>
#include <opencv2/opencv.hpp>


#include "Layers.h"
#include "../Tensor.h"

using namespace cv;

class ConvoLayer : public Layers{
private:
	int numFilters, kernelSize, padding, batchSize;
	int inputHeight, inputWidth, inputChannels;
	int outputHeight, outputWidth;
	int t;
	std::pair<int, int> strides;
	std::string activation;
	bool regularization;
	Eigen::MatrixXd WGrad, vdw, sdw;

	std::vector<std::vector<Eigen::MatrixXd>> WOld; // (numFilters, channels, h, w)
	std::vector<std::vector<Eigen::MatrixXd>> layerOutput, outputGradients, nodeGrads; //(batch_size, numFilters, outputHeight, outputWidth)
	Eigen::VectorXd BGradients, vdb, sdb; //(numFilters)
	std::vector<std::vector<Eigen::MatrixXd>> x; // (batch_size, channels, h, w)


	//testing
	std::vector<Eigen::MatrixXd> XMat, XGrad; //(batch_size, kernelSize * kernelSize * channels, out_h * out_w)
	std::vector<Eigen::MatrixXd> nodeGradsM; //(batch_size, numFilters, out_h * out_w)
	std::string name;
	//testing

	//testing
	std::vector<Mat> outputCv;
	//testing
public:

	ConvoLayer(int numFilters, int kernelSize, std::pair<int, int> strides, int padding, std::string activation, std::string name, bool regularization = false);

	std::unordered_map<std::string, int> initSizes(std::unordered_map<std::string, int>& sizes) override;

	void customInit(const Eigen::MatrixXd& wInput, const Eigen::VectorXd& bInput);

	Tensor forward(const Tensor& inputTensor) override;

	Tensor backward(const Tensor& dyTensor) override;

	void gradientDescent(double alpha) override;

	void saveWeights(const std::string& filename) override;

	void loadWeights(const std::string& filename) override;

	//testing
	void addStuff(std::vector<double>& dO) override;
	//testing
};
