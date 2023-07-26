#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <algorithm>
#include <utility>

#include "ConvoLayer.h"

ConvoLayer::ConvoLayer(int numFilters, int kernelSize, std::pair<int, int> strides, int padding, std::string activation) : numFilters(numFilters), kernelSize(kernelSize), strides(strides), activation(activation), padding(padding) {
	b = Eigen::VectorXd(numFilters);
	BGradients = Eigen::VectorXd(numFilters);
}

std::unordered_map<std::string, int> ConvoLayer::initSizes(std::unordered_map<std::string, int> sizes) {
	int inputChannels = sizes["input channels"];
	int inputHeight = sizes["input height"];
	int inputWidth = sizes["input width"];
	batchSize = sizes["batch size"];

	int outputHeight = (inputHeight - kernelSize + 2 * padding) / strides.first + 1;
	int outputWidth = (inputWidth - kernelSize + 2 * padding) / strides.second + 1;
	W = std::vector<std::vector<Eigen::MatrixXd>>(numFilters, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd::Random(kernelSize, kernelSize)));
	WGradients = std::vector<std::vector<Eigen::MatrixXd>>(numFilters, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd::Zero(kernelSize, kernelSize)));
	layerOutput = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(numFilters, Eigen::MatrixXd(outputHeight, outputWidth)));
	nodeGrads = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(numFilters, Eigen::MatrixXd(outputHeight, outputWidth)));
	x = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd(inputHeight + 2 * padding, inputWidth + 2 * padding)));
	outputGradients = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd(inputHeight, inputWidth)));

	// output sizes
	std::unordered_map<std::string, int> outputSizes;
	outputSizes["input channels"] = numFilters;
	outputSizes["input height"] = outputHeight;
	outputSizes["input width"] = outputWidth;
	outputSizes["batch size"] = batchSize;
	return outputSizes;
}

Tensor ConvoLayer::forward(Tensor inputTensor){

	std::vector<std::vector<Eigen::MatrixXd>> input = inputTensor.matrix4d;

	// add the padding
	for (int z = 0; z < batchSize; z++) {
		for (int c = 0; c < input[0].size(); c++) {
			x[z][c].setZero();  // Set the entire matrix to zero
			x[z][c].block(padding, padding, input[0][0].rows(), input[0][0].cols()) = input[z][c];
		}
	}


	for (int z = 0; z < batchSize; z++) {
		for (int f = 0; f < W.size(); f++) {
			for (int i = 0; i < layerOutput[0][0].rows(); i++) {
				for (int j = 0; j < layerOutput[0][0].cols(); j++) {
					int ii = i * strides.first;
					int jj = j * strides.second;

					double dotP = 0.0;
					for (int c = 0; c < x[0].size(); c++) {
						Eigen::Map<Eigen::VectorXd> v1(W[f][c].data(), W[f][c].size());
						Eigen::Map<Eigen::VectorXd> v2(x[z][c].block(ii, jj, kernelSize, kernelSize).data(), kernelSize * kernelSize);
						dotP += v1.dot(v2);
					}

					dotP += b[f];
					// apply activation function (relu in this case)
					nodeGrads[z][f](i, j) = dotP > 0 ? 1 : 0;
					if (activation == "relu") dotP = std::max(0.0, dotP);
					layerOutput[z][f](i, j) = dotP;
				}
			}
		}
	}

	// <----------------Calculate the nodeGrad------------------>
	


	return Tensor::tensorWrap(layerOutput);
}

Tensor ConvoLayer::backward(Tensor dyTensor) {

	std::vector<std::vector<Eigen::MatrixXd>> dy = dyTensor.matrix4d;

	// Apply activation gradient
	for (int z = 0; z < dy.size(); z++) {
		for (int f = 0; f < dy[0].size(); f++) {
			dy[z][f] = dy[z][f].array() * nodeGrads[z][f].array();
		}
	}


	// Calculate WGradient
	for (int z = 0; z < batchSize; z++) {

		for (int f = 0; f < W.size(); f++) {
			for (int c = 0; c < W[0].size(); c++) {

				for (int i = 0; i < kernelSize; i++) {
					for (int j = 0; j < kernelSize; j++) {

						Eigen::Map<Eigen::VectorXd> v1(dy[z][f].data(), dy[z][f].size());
						Eigen::Map<Eigen::VectorXd> v2(x[z][c].block(i, j, kernelSize, kernelSize).data(), kernelSize * kernelSize);
						WGradients[f][c](i, j) += v1.dot(v2);

					}
				}
			}

			//  Calculate BGradient
			BGradients[f] += dy[z][f].sum();
		}
	}


	// Calculate output gradient
	for (int z = 0; z < batchSize; z++) {
		for (int c = 0; c < W[0].size(); c++) {

			for (int f = 0; f < W.size(); f++) {

				for (int i = 0; i < layerOutput[0][0].rows(); i++) {
					for (int j = 0; j < layerOutput[0][0].cols(); j++) {
						int ii = i * strides.first;
						int jj = j * strides.second;

						for (int iw = 0; iw < kernelSize; iw++) {
							for (int jw = 0; jw < kernelSize; jw++) {
								outputGradients[z][c](ii + iw, jj + jw) += dy[z][c](i, j) * W[f][c](iw, jw);
							}
						}
					}
				}
			}
		}
	}

	return Tensor::tensorWrap(outputGradients);
}
