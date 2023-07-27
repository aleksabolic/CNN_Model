#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <utility>
#include <omp.h>
#include <fstream>


#include "ConvoLayer.h"

ConvoLayer::ConvoLayer(int numFilters, int kernelSize, std::pair<int, int> strides, int padding, std::string activation) : numFilters(numFilters), kernelSize(kernelSize), strides(strides), activation(activation), padding(padding) {
	b = Eigen::VectorXd(numFilters);
	BGradients = Eigen::VectorXd(numFilters);
	trainable = true;
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

Tensor ConvoLayer::forward(Tensor inputTensor) {

	std::vector<std::vector<Eigen::MatrixXd>> input = inputTensor.matrix4d;

	// add the padding
	if (padding) {
		#pragma omp parallel for
		for (int z = 0; z < batchSize; z++) {
			for (int c = 0; c < input[0].size(); c++) {
				x[z][c].setZero();  // Set the entire matrix to zero
				x[z][c].block(padding, padding, input[0][0].rows(), input[0][0].cols()) = input[z][c];
			}
		}
	}
	else {
		x = input;
	}
		
	#pragma omp parallel for
	for (int z = 0; z < batchSize; z++) {
		#pragma omp parallel for
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
	#pragma omp parallel for
	for (int z = 0; z < batchSize; z++) {
		#pragma omp parallel for
		for (int f = 0; f < W.size(); f++) {
			for (int c = 0; c < W[0].size(); c++) {

				for (int i = 0; i < kernelSize; i++) {
					for (int j = 0; j < kernelSize; j++) {

						Eigen::Map<Eigen::VectorXd> v1(dy[z][f].data(), dy[z][f].size());
						Eigen::Map<Eigen::VectorXd> v2(x[z][c].block(i, j, dy[z][f].rows(), dy[z][f].cols()).data(), dy[z][f].size());
						WGradients[f][c](i, j) += v1.dot(v2);

					}
				}
			}

			//  Calculate BGradient
			BGradients[f] += dy[z][f].sum();
		}
	}


	// Calculate output gradient
	#pragma omp parallel for
	for (int z = 0; z < batchSize; z++) {
		#pragma omp parallel for
		for (int c = 0; c < W[0].size(); c++) {
			#pragma omp parallel for
			for (int f = 0; f < W.size(); f++) {

				for (int i = 0; i < layerOutput[0][0].rows(); i++) {
					for (int j = 0; j < layerOutput[0][0].cols(); j++) {
						int ii = i * strides.first;
						int jj = j * strides.second;

						outputGradients[z][c].block(ii,jj,kernelSize, kernelSize) += dy[z][f](i, j) * W[f][c];
					}
				}
			}
		}
	}

	return Tensor::tensorWrap(outputGradients);
}

void ConvoLayer::gradientDescent(double alpha) {
	for (int f = 0; f < W.size(); f++) {
		for (int c = 0; c < W[0].size(); c++) {
			W[f][c] -= alpha * WGradients[f][c];
			WGradients[f][c].setZero();
		}
		b[f] -= alpha * BGradients[f];
		BGradients[f] = 0;
	}
}

void ConvoLayer::saveWeights(const std::string& filename) {
	std::ofstream outfile(filename, std::ios::binary);

	for (const auto& w_layer : W) {
		for (const auto& w_matrix : w_layer) {
			Eigen::MatrixXd::Index rows = w_matrix.rows(), cols = w_matrix.cols();
			outfile.write((char*)(&rows), sizeof(Eigen::MatrixXd::Index));
			outfile.write((char*)(&cols), sizeof(Eigen::MatrixXd::Index));
			outfile.write((char*)w_matrix.data(), rows * cols * sizeof(Eigen::MatrixXd::Scalar));
		}
	}

	Eigen::VectorXd::Index b_size = b.size();
	outfile.write((char*)(&b_size), sizeof(Eigen::VectorXd::Index));
	outfile.write((char*)b.data(), b_size * sizeof(Eigen::VectorXd::Scalar));

	outfile.close();
}

void ConvoLayer::loadWeights(const std::string& filename) {
	std::ifstream infile(filename, std::ios::binary);

	Eigen::MatrixXd::Index rows = 0, cols = 0;

	for (auto& filter : W) {
		for (auto& w_matrix : filter) {
			infile.read((char*)(&rows), sizeof(Eigen::MatrixXd::Index));
			infile.read((char*)(&cols), sizeof(Eigen::MatrixXd::Index));
			w_matrix = Eigen::MatrixXd(rows, cols);
			infile.read((char*)w_matrix.data(), rows * cols * sizeof(Eigen::MatrixXd::Scalar));
		}
	}

	Eigen::VectorXd::Index b_size = 0;
	infile.read((char*)(&b_size), sizeof(Eigen::VectorXd::Index));
	b = Eigen::VectorXd(b_size);
	infile.read((char*)b.data(), b_size * sizeof(Eigen::VectorXd::Scalar));

	infile.close();
}
