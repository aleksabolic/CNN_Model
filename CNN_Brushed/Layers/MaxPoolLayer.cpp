#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <iostream>
#include <omp.h>

#include "MaxPoolLayer.h"

MaxPoolLayer::MaxPoolLayer(int kernelSize, int stride) : kernelSize(kernelSize), stride(stride) {
	trainable = false;
}

std::unordered_map<std::string, int> MaxPoolLayer::initSizes(std::unordered_map<std::string, int>& sizes) {
	int inputChannels = sizes["input channels"];
	int inputHeight = sizes["input height"];
	int inputWidth = sizes["input width"];
	batchSize = sizes["batch size"];

	int outputHeight = (inputHeight - kernelSize) / stride + 1;
	int outputWidth = (inputWidth - kernelSize) / stride + 1;
	layerOutput = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd(outputHeight, outputWidth)));
	gradGate = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd::Zero(inputHeight, inputWidth)));
	outputGradients = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd(inputHeight, inputWidth)));
	//x = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd(inputHeight, inputWidth)));

	std::unordered_map<std::string, int> outputSizes;
	outputSizes["input channels"] = inputChannels;
	outputSizes["input height"] = outputHeight;
	outputSizes["input width"] = outputWidth;
	outputSizes["batch size"] = batchSize;
	return outputSizes;
}

Tensor MaxPoolLayer::forward(const Tensor& inputTensor) {

	std::vector<std::vector<Eigen::MatrixXd>> input = inputTensor.matrix4d;

	int outputRows = layerOutput[0][0].rows();
	int outputCols = layerOutput[0][0].cols();

	#pragma omp parallel for
	for (int z = 0; z < batchSize; z++) {
		#pragma omp parallel for
		for (int c = 0; c < input[0].size(); c++) {

			// Refresh the gradient gate
			gradGate[z][c].setZero();

			for (int i = 0; i < outputRows; i++) {
				for (int j = 0; j < outputCols; j++) {
					int ii = i * stride;
					int jj = j * stride;

					// Select the sub-matrix for pooling
					Eigen::MatrixXd subMatrix = input[z][c].block(ii, jj, kernelSize, kernelSize);

					// Find the max value and its index
					Eigen::MatrixXd::Index maxRow, maxCol;
					double maxVal = subMatrix.maxCoeff(&maxRow, &maxCol);

					// Update the output and gradient
					layerOutput[z][c](i, j) = maxVal;
					gradGate[z][c](ii + maxRow, jj + maxCol) = 1;
				}
			}
		}
	}

	return Tensor::tensorWrap(layerOutput);
}

Tensor MaxPoolLayer::backward(const Tensor& dyTensor) {
	std::vector<std::vector<Eigen::MatrixXd>> dy = dyTensor.matrix4d;
	#pragma omp parallel for
	for (int z = 0; z < dy.size(); z++) {
		#pragma omp parallel for
		for (int c = 0; c < dy[0].size(); c++) {

			for (int i = 0; i < dy[0][0].rows(); i++) {
				for (int j = 0; j < dy[0][0].cols(); j++) {

					int ii = i * stride;
					int jj = j * stride;

					gradGate[z][c].block(ii, jj, kernelSize, kernelSize) *= dy[z][c](i, j);				
				}
			}

		}
	}
	return Tensor::tensorWrap(gradGate);
}

void MaxPoolLayer::gradientDescent(double alpha) {
	// Nothing to do here
}

void MaxPoolLayer::saveWeights(const std::string& filename) {
	// Nothing to do here
}

void MaxPoolLayer::loadWeights(const std::string& filename) {
	// Nothing to do here
}