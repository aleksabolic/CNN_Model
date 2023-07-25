#include <vector>
#include <unordered_map>
#include <Eigen/Dense>

#include "MaxPoolLayer.h"

MaxPoolLayer::MaxPoolLayer(int kernelSize, int stride) : kernelSize(kernelSize), stride(stride) {}

std::unordered_map<std::string, int> MaxPoolLayer::initSizes(std::unordered_map<std::string, int> sizes) {
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

Tensor MaxPoolLayer::forward(Tensor inputTensor) {

	std::vector<std::vector<Eigen::MatrixXd>> input = inputTensor.matrix4d;

	for (int z = 0; z < batchSize; z++) {
		for (int c = 0; c < input[0].size(); c++) {

			for (int i = 0; i < layerOutput[0][0].rows(); i++) {
				for (int j = 0; j < layerOutput[0][0].cols(); j++) {
					int ii = i * stride;
					int jj = j * stride;

					// find the max pixel in the input channel
					double maxVal = -DBL_MAX;
					int maxII, maxJJ;
					for (int relativeI = 0; relativeI < kernelSize; relativeI++) {
						for (int relativeJ = 0; relativeJ < kernelSize; relativeJ++) {
							if (input[z][c](ii + relativeI, jj + relativeJ) > maxVal) {
								maxVal - input[z][c](ii + relativeI, jj + relativeJ);
								maxII = ii + relativeI;
								maxJJ = jj + relativeJ;
							}
						}
					}
					gradGate[z][c](maxII, maxJJ) = 1;
					layerOutput[z][c](i, j) = maxVal;
				}
			}
		}
	}
	return Tensor::tensorWrap(layerOutput);
}

Tensor MaxPoolLayer::backward(Tensor dyTensor) {
	std::vector<std::vector<Eigen::MatrixXd>> dy = dyTensor.matrix4d;
	for (int z = 0; z < dy.size(); z++) {
		for (int c = 0; c < dy[0].size(); c++) {

			for (int i = 0; i < dy[0][0].rows(); i++) {
				for (int j = 0; j < dy[0][0].cols(); j++) {
					int ii = i * stride;
					int jj = j * stride;

					for (int relativeI = 0; relativeI < kernelSize; relativeI++) {
						for (int relativeJ = 0; relativeJ < kernelSize; relativeJ++) {
							gradGate[z][c](ii + relativeI, jj + relativeJ) *= dy[z][c](i, j);
						}
					}
				}
			}
		}
	}
	return Tensor::tensorWrap(gradGate);
}
