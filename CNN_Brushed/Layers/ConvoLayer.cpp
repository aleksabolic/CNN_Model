#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <utility>
#include <omp.h>
#include <fstream>
#include <random>

#include "ConvoLayer.h"

ConvoLayer::ConvoLayer(int numFilters, int kernelSize, std::pair<int, int> strides, int padding, std::string activation, bool regularization) : numFilters(numFilters), kernelSize(kernelSize), strides(strides), activation(activation), padding(padding), regularization(regularization) {
	b = Eigen::VectorXd(numFilters);
	vdb = Eigen::VectorXd::Zero(numFilters);
	sdb = Eigen::VectorXd::Zero(numFilters);
	BGradients = Eigen::VectorXd::Zero(numFilters);
	t = 1;
	trainable = true;
}

std::unordered_map<std::string, int> ConvoLayer::initSizes(std::unordered_map<std::string, int>& sizes) {

	int inputChannels = sizes["input channels"];
	int inputHeight = sizes["input height"];
	int inputWidth = sizes["input width"];
	batchSize = sizes["batch size"];

	int outputHeight = (inputHeight - kernelSize + 2 * padding) / strides.first + 1;
	int outputWidth = (inputWidth - kernelSize + 2 * padding) / strides.second + 1;
	//testing
	W = Eigen::MatrixXd(numFilters, kernelSize * kernelSize * inputChannels);
	wgt = Eigen::MatrixXd::Zero(numFilters, kernelSize * kernelSize * inputChannels);
	vdw = Eigen::MatrixXd::Zero(numFilters, kernelSize * kernelSize * inputChannels);
	sdw = Eigen::MatrixXd::Zero(numFilters, kernelSize * kernelSize * inputChannels);
	//testing
	WOld = std::vector<std::vector<Eigen::MatrixXd>>(numFilters, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd(kernelSize, kernelSize)));
	WGradients = std::vector<std::vector<Eigen::MatrixXd>>(numFilters, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd::Zero(kernelSize, kernelSize)));
	layerOutput = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(numFilters, Eigen::MatrixXd(outputHeight, outputWidth)));
	nodeGrads = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(numFilters, Eigen::MatrixXd(outputHeight, outputWidth)));
	x = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd(inputHeight + 2 * padding, inputWidth + 2 * padding)));
	outputGradients = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd::Zero(inputHeight, inputWidth)));

	std::random_device rd{};
	std::mt19937 gen{rd()};
	double std_dev = sqrt(2.0 / (inputChannels * kernelSize * kernelSize)); // He init for convolutional layers
	std::normal_distribution<> d{0, std_dev}; // Mean 0, standard deviation calculated by He initialization

	//testing
	for (int i = 0; i < W.rows(); ++i) {
		for (int j = 0; j < W.cols(); ++j) {
			W(i, j) = d(gen);
		}
	}
		//init WOld
	for (int f = 0; f < numFilters; f++) {
		for (int c = 0; c < inputChannels; c++) {
			Eigen::RowVectorXd v1 = W.block(f, c * kernelSize * kernelSize, 1, kernelSize * kernelSize);
			WOld[f][c] = Eigen::Map<Eigen::MatrixXd>(v1.data(), kernelSize, kernelSize);
		}
	}
	//testing

	for (int i = 0; i < b.size(); ++i) {
		b(i) = d(gen);
	}


	// output sizes
	std::unordered_map<std::string, int> outputSizes;
	outputSizes["input channels"] = numFilters;
	outputSizes["input height"] = outputHeight;
	outputSizes["input width"] = outputWidth;
	outputSizes["batch size"] = batchSize;

	//print outptu sizes
	std::cout << "---------------------convo layer output sizes --------------------------------\n";
	std::cout << "output channels: " << outputSizes["input channels"] << std::endl;
	std::cout << "output height: " << outputSizes["input height"] << std::endl;
	std::cout << "output width: " << outputSizes["input width"] << std::endl;
	std::cout << "batch size: " << outputSizes["batch size"] << std::endl;
	std::cout << "---------------------convo layer output sizes --------------------------------\n";

	return outputSizes;
}

//Tensor ConvoLayer::forward(const Tensor& inputTensor) {
//
//	std::vector<std::vector<Eigen::MatrixXd>> input = inputTensor.matrix4d;
//
//	/*std::cout << "---------------------convo layer input --------------------------------\n";
//	Tensor::tensorWrap(input).print();
//	std::cout << "-------convo layer WWWWWW --------\n";
//	std::cout << W << std::endl;
//	std::cout << "-------convo layer WWWWWW --------\n";*/
//
//
//	// add the padding
//	if (padding) {
//		#pragma omp parallel for
//		for (int z = 0; z < batchSize; z++) {
//			for (int c = 0; c < input[0].size(); c++) {
//				x[z][c].setZero();  // Set the entire matrix to zero
//				x[z][c].block(padding, padding, input[0][0].rows(), input[0][0].cols()) = input[z][c];
//			}
//		}
//	}
//	else {
//		x = input;
//	}
//
//
//	#pragma omp parallel for
//	for (int z = 0; z < batchSize; z++) {
//		int inputChannels = x[0].size();
//		Eigen::MatrixXd X = Eigen::MatrixXd(kernelSize * kernelSize * inputChannels, layerOutput[0][0].size());
//
//		// filling the X matrix
//		for (int i = 0; i < layerOutput[0][0].rows(); i++) {
//			for (int j = 0; j < layerOutput[0][0].cols(); j++) {
//				int ii = i * strides.first;
//				int jj = j * strides.second;
//
//				for (int c = 0; c < inputChannels; c++) {
//					Eigen::Map<Eigen::VectorXd> v1(x[z][c].block(ii, jj, kernelSize, kernelSize).data(), kernelSize * kernelSize);
//					X.block(c * kernelSize * kernelSize, i * layerOutput[0][0].cols() + j, kernelSize * kernelSize, 1) = v1;
//				}
//			}
//		}
//
//		// calculate the output
//		Eigen::MatrixXd output = W * X;
//
//		// add the bias
//		output = output.colwise() + b;
//
//		for (int f = 0; f < numFilters; f++) {
//			for (int i = 0; i < layerOutput[0][0].rows(); i++) {
//				for (int j = 0; j < layerOutput[0][0].cols(); j++) {
//					double dotP = output(f, i * layerOutput[0][0].cols() + j);
//					// apply activation function (relu in this case)
//					if (activation == "relu") {
//						nodeGrads[z][f](i, j) = dotP > 0 ? 1 : 0;
//						dotP = std::max(0.0, dotP);
//					}
//					// Adding Leaky ReLU activation
//					else if (activation == "leaky_relu") {
//						nodeGrads[z][f](i, j) = dotP > 0 ? 1 : 0.01;
//						dotP = dotP > 0 ? dotP : 0.01 * dotP;
//					}
//					else if (activation == "linear") {
//						nodeGrads[z][f](i, j) = 1;
//					}
//					layerOutput[z][f](i, j) = dotP;
//				}
//
//			}
//		}
//	}
//
//	return Tensor::tensorWrap(layerOutput);
//}

Tensor ConvoLayer::forward(const Tensor& inputTensor) {
	//convert w to wold
	for (int f = 0; f < numFilters; f++) {
		for (int c = 0; c < x[0].size(); c++) {
			Eigen::RowVectorXd v1 = W.block(f, c * kernelSize * kernelSize, 1, kernelSize * kernelSize);
			WOld[f][c] = Eigen::Map<Eigen::MatrixXd>(v1.data(), kernelSize, kernelSize);
		}
	}


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
		for (int f = 0; f < WOld.size(); f++) {
			for (int i = 0; i < layerOutput[0][0].rows(); i++) {
				for (int j = 0; j < layerOutput[0][0].cols(); j++) {
					int ii = i * strides.first;
					int jj = j * strides.second;
					double dotP = 0.0;
					for (int c = 0; c < x[0].size(); c++) {
						//Eigen::Map<Eigen::VectorXd> v1(WOld[f][c].data(), WOld[f][c].size());
						//Eigen::Map<Eigen::VectorXd> v2(x[z][c].block(ii, jj, kernelSize, kernelSize).data(), kernelSize * kernelSize);
						dotP +=( WOld[f][c].array() * x[z][c].block(ii, jj, kernelSize, kernelSize).array()).sum();
					}
					dotP += b(f);
					// apply activation function (relu in this case)
					if (activation == "relu") {
						nodeGrads[z][f](i, j) = dotP > 0 ? 1 : 0;
						dotP = std::max(0.0, dotP);
					}
					// Adding Leaky ReLU activation
					else if (activation == "leaky_relu") {
						nodeGrads[z][f](i, j) = dotP > 0 ? 1 : 0.01;
						dotP = dotP > 0 ? dotP : 0.01 * dotP;
					}
					else if (activation == "linear") {
						nodeGrads[z][f](i, j) = 1;
					}
					layerOutput[z][f](i, j) = dotP;
				}
			}
		}
	}
	return Tensor::tensorWrap(layerOutput);
}

Tensor ConvoLayer::backward(const Tensor& dyTensor) {

	std::vector<std::vector<Eigen::MatrixXd>> dy = dyTensor.matrix4d;
	/*std::cout<<"---------------------dy--------------------------------\n";
	Tensor::tensorWrap(dy).print();
	std::cout << "---------------------dy--------------------------------\n";*/

	// Apply activation gradient
	for (int z = 0; z < dy.size(); z++) {
		for (int f = 0; f < dy[0].size(); f++) {
			dy[z][f] = dy[z][f].array() * nodeGrads[z][f].array();
		}
	}

	// Calculate WGradient
	//#pragma omp parallel for
	//for (int f = 0; f < numFilters; f++) {
	//	int inputChannels = x[0].size();
	//	int outputHeight = layerOutput[0][0].rows();
	//	int outputWidth = layerOutput[0][0].cols();
	//	int outputSize = outputHeight * outputWidth;

	//	Eigen::VectorXd DY = Eigen::VectorXd(batchSize * outputSize);
	//	// Filling the DY vector
	//	for (int z = 0; z < batchSize; z++) {
	//		DY.segment(z * outputSize, outputSize) = Eigen::VectorXd::Map(dy[z][f].data(), outputSize);
	//	}

	//	#pragma omp parallel for
	//	for (int i = 0; i < kernelSize; i++) {
	//		for (int j = 0; j < kernelSize; j++) {

	//			Eigen::MatrixXd X(inputChannels, batchSize * outputSize);
	//			// Filling the X matrix
	//			for (int c = 0; c < inputChannels; ++c) {
	//				for (int z = 0; z < batchSize; ++z) {
	//					X.row(c).segment(z * outputSize, outputSize) = Eigen::Map<Eigen::RowVectorXd>(x[z][c].block(i, j, outputHeight, outputWidth).data(), outputSize);
	//				}
	//			}

	//			Eigen::MatrixXd WGradientrow = X * DY;
	//			for (int c = 0; c < inputChannels; c++) {
	//				WGradients[f][c](i, j) = WGradientrow(c,0);
	//			}
	//		}
	//	}
	//	for (int z = 0; z < batchSize; z++) {
	//		BGradients[f] += dy[z][f].sum();
	//	}
	//}
	
	// Reset gradients
	/*for (int f = 0; f < WOld.size(); f++) {
		for (int c = 0; c < WOld[0].size(); c++) {
			WGradients[f][c].setZero();
		}
	}
	BGradients.setZero(); */



	// Calculate WGradient
	#pragma omp parallel for
	for (int z = 0; z < batchSize; z++) {
	#pragma omp parallel for
		for (int f = 0; f < WOld.size(); f++) {
			for (int c = 0; c < WOld[0].size(); c++) {
				for (int i = 0; i < kernelSize; i++) {
					for (int j = 0; j < kernelSize; j++) {
						WGradients[f][c](i,j) += (x[z][c].block(i, j, dy[z][f].rows(), dy[z][f].cols()).array() * dy[z][f].array()).sum();
					}
				}
			}
			//  Calculate BGradient
			BGradients[f] += dy[z][f].sum();
		}
	}


	/*for (int z = 0; z < batchSize; z++) {
		for (int f = 0; f < WOld.size(); f++) {
			for (int c = 0; c < WOld[0].size(); c++) {
				for (int i = 0; i < dy[0][0].rows(); i++) {
					for (int j = 0; j < dy[0][0].cols(); j++) {
						int ii = i * strides.first;
						int jj = j * strides.second;

						WGradients[f][c] += x[z][c].block(ii, jj, kernelSize, kernelSize) * dy[z][f](i, j);
					}
				}
			}
			BGradients[f] += dy[z][f].sum();
		}
	}*/

	// reset the output gradients
	for (int z = 0; z < batchSize; z++) {
		for (int c = 0; c < x[0].size(); c++) {
			outputGradients[z][c].setZero();
		}
	}

	// Calculate output gradient
	#pragma omp parallel for
	for (int z = 0; z < batchSize; z++) {
		for (int c = 0; c < x[0].size(); c++) {
			for (int f = 0; f < numFilters; f++) {
				Eigen::MatrixXd dyf = dy[z][f];
				for (int i = 0; i < kernelSize; i++) {
					for (int j = 0; j < kernelSize; j++) {
						//<------add the stride------>
						outputGradients[z][c].block(i, j, dyf.rows(), dyf.cols()) += dyf * WOld[f][c](i, j);
					}
				}
			}
		}
	}

	//// kernel flipping 
	//int dyPad = kernelSize - 1;
	//std::vector<std::vector<Eigen::MatrixXd>> dyPadded = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(dy[0].size(), Eigen::MatrixXd::Zero(dy[0][0].rows() + 2 * dyPad, dy[0][0].cols() + 2 * dyPad)));
	////add the padding to the dy
	//#pragma omp parallel for
	//for (int z = 0; z < batchSize; z++) {
	//	for (int f = 0; f < dy[0].size(); f++) {
	//		dyPadded[z][f].block(dyPad, dyPad, dy[0][0].rows(), dy[0][0].cols()) = dy[z][f];
	//	}
	//}

	//for (int z = 0; z < batchSize; z++) {
	//	for (int c = 0; c < x[0].size(); c++) {
	//		Eigen::MatrixXd accumulatedGradient = Eigen::MatrixXd::Zero(x[0][0].rows(), x[0][0].cols());
	//		for (int f = 0; f < numFilters; f++) {
	//			Eigen::MatrixXd flippedKernel = WOld[f][c].colwise().reverse().rowwise().reverse();

	//			Eigen::MatrixXd dyf = dyPadded[z][f];

	//			for (int i = 0; i < x[0][0].rows(); i++) {
	//				for (int j = 0; j < x[0][0].cols(); j++) {
	//					accumulatedGradient(i,j) += (dyf.block(i,j,kernelSize,kernelSize).array() * flippedKernel.array()).sum();
	//				}
	//			}
	//		}
	//		outputGradients[z][c] = accumulatedGradient;
	//	}
	//}

	return Tensor::tensorWrap(outputGradients);
}

// Gradient descent with momentum
void ConvoLayer::gradientDescent(double alpha) {
	double beta1 = 0.9;
	double beta2 = 0.999;
	double epsilon = 1e-8;

	//check if the layer should be regularized
	if (true) {
		std::string regularization = "l2";
		double lambda = 0.01;
		if (regularization == "l2") {
			for (int f = 0; f < WOld.size(); f++) {
				for (int c = 0; c < WOld[0].size(); c++) {
					WGradients[f][c] += lambda * WOld[f][c] / batchSize;
				}
			}
		}
	}

	// filling the wgt matrix
	int inputChannels = x[0].size();
	for (int f = 0; f < numFilters; f++) {
		for (int c = 0; c < inputChannels; c++) {
			Eigen::Map<Eigen::RowVectorXd> v1(WGradients[f][c].data(), WGradients[f][c].size());
			wgt.block(f, c * kernelSize * kernelSize, 1, kernelSize * kernelSize) = v1;
		}
	}

	//Gradient clipping
	/*double maxNorm = 3;
	double norm = wgt.norm();
	if (norm > maxNorm) {
		wgt = wgt * (maxNorm / norm);
	}*/


	vdw = beta1 * vdw + (1 - beta1) * wgt;
	vdb = beta1 * vdb + (1 - beta1) * BGradients;

	sdw = beta2 * sdw + (1 - beta2) * wgt.cwiseProduct(wgt);
	sdb = beta2 * sdb + (1 - beta2) * BGradients.cwiseProduct(BGradients);

	//bias correction
	Eigen::MatrixXd vdwCorr = vdw / (1 - pow(beta1, t));
	Eigen::VectorXd vdbCorr = vdb / (1 - pow(beta1, t));

	Eigen::MatrixXd sdwCorr = sdw / (1 - pow(beta2, t));
	Eigen::VectorXd sdbCorr = sdb / (1 - pow(beta2, t));

	W = W - (alpha * vdwCorr).cwiseQuotient(sdwCorr.unaryExpr([](double x) { return sqrt(x) + 1e-8; }));
	b = b - (alpha * vdbCorr).cwiseQuotient(sdbCorr.unaryExpr([](double x) { return sqrt(x) + 1e-8; }));

	// Convert W to WOld
	for (int f = 0; f < numFilters; f++) {
		for (int c = 0; c < inputChannels; c++) {
			Eigen::RowVectorXd v1 = W.block(f, c * kernelSize * kernelSize, 1, kernelSize * kernelSize);
			WOld[f][c] = Eigen::Map<Eigen::MatrixXd>(v1.data(), kernelSize, kernelSize);
		}
	}

	// Reset gradients
	for (int f = 0; f < WOld.size(); f++) {
		for (int c = 0; c < WOld[0].size(); c++) {
			WGradients[f][c].setZero();
			wgt.setZero();
		}	
	}
	BGradients.setZero();

	t++;
}

void ConvoLayer::saveWeights(const std::string& filename) {
	std::ofstream outfile(filename, std::ios::binary);

	int outer_size = WOld.size();
	outfile.write((char*)&outer_size, sizeof(int));
	
	for (const auto& inner_vector : WOld) {
		int inner_size = inner_vector.size();
		outfile.write((char*)&inner_size, sizeof(int));
		
		for (const auto& matrix : inner_vector) {
			int rows = matrix.rows();
			int cols = matrix.cols();
			outfile.write((char*)&rows, sizeof(int));
			outfile.write((char*)&cols, sizeof(int));
			outfile.write((char*)matrix.data(), rows*cols*sizeof(double));
		}
	}

	int size = b.size();
	outfile.write((char*)&size, sizeof(int));
	outfile.write((char*)b.data(), size*sizeof(double));

	outfile.close();
}

void ConvoLayer::loadWeights(const std::string& filename) {
	std::ifstream infile(filename, std::ios::binary);

	int outer_size;
	infile.read((char*)&outer_size, sizeof(int));

	for (auto& inner_vector : WOld) {
		int inner_size;
		infile.read((char*)&inner_size, sizeof(int));

		for (auto& matrix : inner_vector) {
			int rows, cols;
			infile.read((char*)&rows, sizeof(int));
			infile.read((char*)&cols, sizeof(int));
			infile.read((char*)matrix.data(), rows * cols * sizeof(double));
		}
	}

	int size;
	infile.read((char*)&size, sizeof(int));
	infile.read((char*)b.data(), size * sizeof(double));

	infile.close();

	// fill the W with the WOld
	for (int f = 0; f < numFilters; f++) {
		for (int c = 0; c < WOld[0].size(); c++) {
			Eigen::Map<Eigen::RowVectorXd> v1(WOld[f][c].data(), WOld[f][c].size());
			W.block(f, c * kernelSize * kernelSize, 1, kernelSize * kernelSize) = v1;
		}
	}
}

void ConvoLayer::addStuff(std::vector<double>& dO) {
	// filling the wgt matrix
	int inputChannels = x[0].size();
	for (int f = 0; f < numFilters; f++) {
		for (int c = 0; c < inputChannels; c++) {
			Eigen::Map<Eigen::RowVectorXd> v1(WGradients[f][c].data(), WGradients[f][c].size());
			wgt.block(f, c * kernelSize * kernelSize, 1, kernelSize * kernelSize) = v1;
		}
	}
	// adding the dw
	for (int i = 0; i < wgt.rows(); i++) {
		for (int j = 0; j < wgt.cols(); j++) {
			dO.push_back(wgt(i, j));
		}
	}
	
	//adding the db
	for (int i = 0; i < BGradients.size(); i++) {
		dO.push_back(BGradients(i));
	}
}