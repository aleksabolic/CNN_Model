#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <unordered_map>
#include "FlattenLayer.h"

FlattenLayer::FlattenLayer() {
	trainable = false;
}

std::unordered_map<std::string, int> FlattenLayer::initSizes(std::unordered_map<std::string, int>& sizes) {
	inputChannels = sizes["input channels"];
	inputHeight = sizes["input height"];
	inputWidth = sizes["input width"];
	batchSize = sizes["batch size"];

	// output sizes
	std::unordered_map<std::string, int> outputSizes;
	outputSizes["batch size"] = batchSize;
	outputSizes["input size"] = inputChannels * inputHeight * inputWidth;
	return outputSizes;
}

Tensor FlattenLayer::forward(const Tensor& inputTensor) {

	std::vector<std::vector<Eigen::MatrixXd>> input = inputTensor.matrix4d;

	Eigen::MatrixXd output = Eigen::MatrixXd(batchSize, input[0].size() * input[0][0].size());

	#pragma omp parallel for
	for (int z = 0; z < input.size(); z++) {
		int index = 0;
		for (int c = 0; c < input[0].size(); c++) {
			Eigen::VectorXd vec = Eigen::Map<Eigen::VectorXd>(input[z][c].data(), input[z][c].size());
			output.row(z).segment(index, vec.size()) = vec;
			index += vec.size();
		}
	}

	return Tensor::tensorWrap(output);
}

Tensor FlattenLayer::backward(const Tensor& dyTensor) {

	Eigen::MatrixXd dy = dyTensor.matrix;

	std::vector<std::vector<Eigen::MatrixXd>> output = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd(inputHeight, inputWidth)));

	#pragma omp parallel for
	for (int z = 0; z < dy.rows(); z++) {
		int matrixSize = inputHeight * inputWidth;
		for (int c = 0; c < inputChannels; c++) {
			Eigen::VectorXd channel = dy.row(z).segment(c * matrixSize, matrixSize);
			output[z][c] = Eigen::Map<const Eigen::MatrixXd>(channel.data(), inputHeight, inputWidth);
		}
	}

	return Tensor::tensorWrap(output);
}

void FlattenLayer::gradientDescent(double alpha) {
	// Nothing to do here
}

void FlattenLayer::saveWeights(const std::string& filename) {
	// Nothing to do here
}

void FlattenLayer::loadWeights(const std::string& filename) {
	// Nothing to do here
}