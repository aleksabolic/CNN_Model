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

ConvoLayer::ConvoLayer(int numFilters, int kernelSize, std::pair<int, int> strides, int padding, std::string activation, std::string name, bool regularization) : numFilters(numFilters), kernelSize(kernelSize), strides(strides), activation(activation), padding(padding), regularization(regularization), name(name) {
	b = Eigen::VectorXd(numFilters);
	vdb = Eigen::VectorXd::Zero(numFilters);
	sdb = Eigen::VectorXd::Zero(numFilters);
	BGradients = Eigen::VectorXd::Zero(numFilters);
	t = 1;
	trainable = true;
}

std::unordered_map<std::string, int> ConvoLayer::initSizes(std::unordered_map<std::string, int>& sizes) {

	inputChannels = sizes["input channels"];
	inputHeight = sizes["input height"];
	inputWidth = sizes["input width"];
	batchSize = sizes["batch size"];

	outputHeight = (inputHeight - kernelSize + 2 * padding) / strides.first + 1;
	outputWidth = (inputWidth - kernelSize + 2 * padding) / strides.second + 1;

	W = Eigen::MatrixXd(numFilters, kernelSize * kernelSize * inputChannels);
	WGrad = Eigen::MatrixXd::Zero(numFilters, kernelSize * kernelSize * inputChannels);
	vdw = Eigen::MatrixXd::Zero(numFilters, kernelSize * kernelSize * inputChannels);
	sdw = Eigen::MatrixXd::Zero(numFilters, kernelSize * kernelSize * inputChannels);
	// WOld is the same as W but in a different format, everytime W changes WOld is updated
	WOld = std::vector<std::vector<Eigen::MatrixXd>>(numFilters, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd(kernelSize, kernelSize)));
	layerOutput = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(numFilters, Eigen::MatrixXd(outputHeight, outputWidth)));
	nodeGrads = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(numFilters, Eigen::MatrixXd(outputHeight, outputWidth)));
	x = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd(inputHeight + 2 * padding, inputWidth + 2 * padding)));
	outputGradients = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd::Zero(inputHeight, inputWidth)));

	//testing
	XMat = std::vector<Eigen::MatrixXd>(batchSize, Eigen::MatrixXd(kernelSize * kernelSize * inputChannels, outputHeight * outputWidth));
	XGrad = std::vector<Eigen::MatrixXd>(batchSize, Eigen::MatrixXd(kernelSize * kernelSize * inputChannels, outputHeight * outputWidth));
	nodeGradsM = std::vector<Eigen::MatrixXd>(batchSize, Eigen::MatrixXd(numFilters, outputHeight * outputWidth));
	//testing

	std::random_device rd{};
	std::mt19937 gen{rd()};
	double std_dev = sqrt(2.0 / (inputChannels * kernelSize * kernelSize)); // He init for convolutional layers
	std::normal_distribution<> d{0, std_dev}; // Mean 0, standard deviation calculated by He initialization

	//init W
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

	for (int i = 0; i < b.size(); ++i) {
		b(i) = d(gen);
	}


	// output sizes
	std::unordered_map<std::string, int> outputSizes;
	outputSizes["input channels"] = numFilters;
	outputSizes["input height"] = outputHeight;
	outputSizes["input width"] = outputWidth;
	outputSizes["batch size"] = batchSize;

	//print output sizes
	std::cout << "---------------------convo layer output sizes --------------------------------\n";
	std::cout << "output channels: " << outputSizes["input channels"] << std::endl;
	std::cout << "output height: " << outputSizes["input height"] << std::endl;
	std::cout << "output width: " << outputSizes["input width"] << std::endl;
	std::cout << "batch size: " << outputSizes["batch size"] << std::endl;
	std::cout << "---------------------convo layer output sizes --------------------------------\n";

	return outputSizes;
}

void ConvoLayer::customInit(const Eigen::MatrixXd& wInput, const Eigen::VectorXd& bInput) {
	if (W.rows() != wInput.rows() || W.cols() != wInput.cols()) {
		std::cout << "Error: customInit: wInput has wrong dimensions" << std::endl;
		if (W.rows() != wInput.rows()) {
			std::cout<< "Rows don't match: " << W.rows() << " != " << wInput.rows() << std::endl;
		}
		if (W.cols() != wInput.cols()) {
			std::cout<< "Cols don't match: " << W.cols() << " != " << wInput.cols() << std::endl;
		}
		return;
	}
	W = wInput;
	//init WOld
	for (int f = 0; f < numFilters; f++) {
		for (int c = 0; c < W.cols() / (kernelSize * kernelSize); c++) {
			Eigen::RowVectorXd v1 = W.block(f, c * kernelSize * kernelSize, 1, kernelSize * kernelSize);
			WOld[f][c] = Eigen::Map<Eigen::MatrixXd>(v1.data(), kernelSize, kernelSize);
		}
	}
	b = bInput;
}

Tensor ConvoLayer::forward(const Tensor& inputTensor) {

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
		//Eigen::MatrixXd X = Eigen::MatrixXd(kernelSize * kernelSize * inputChannels, layerOutput[0][0].size());

		// filling the X matrix
		for (int i = 0; i < layerOutput[0][0].rows(); i++) {
			for (int j = 0; j < layerOutput[0][0].cols(); j++) {
				int ii = i * strides.first;
				int jj = j * strides.second;

				for (int c = 0; c < inputChannels; c++) {
					Eigen::Map<Eigen::VectorXd> v1(x[z][c].block(ii, jj, kernelSize, kernelSize).data(), kernelSize * kernelSize);
					//X.block(c * kernelSize * kernelSize, i * layerOutput[0][0].cols() + j, kernelSize * kernelSize, 1) = v1;
					XMat[z].block(c * kernelSize * kernelSize, i * layerOutput[0][0].cols() + j, kernelSize * kernelSize, 1) = v1;

				}
			}
		}

		// calculate the output (one output of the batch)
		//Eigen::MatrixXd output = W * X;
		Eigen::MatrixXd output = W * XMat[z];

		// add the bias
		output = output.colwise() + b;

#pragma region Activation
		// APPLY ACTIVATION AND CALC NODE GRADS

		auto relu = [](double x) {return x > 0 ? x : 0.0; };
		auto reluGrad = [](double x) {return x > 0 ? 1.0 : 0.0; };

		auto exponent = [](double x) {return exp(x); };

		auto softmaxFix = [](double x) {return std::max(x, 1e-9); };

		auto leakyRelu = [](double x) {return x > 0 ? x : 0.01 * x; };
		auto leakyReluGrad = [](double x) {return x > 0 ? 1.0 : 0.01; };
		// apply activation function
		if (activation == "relu") {
			nodeGradsM[z] = output.unaryExpr(reluGrad);
			output = output.unaryExpr(relu);
		}
		else if (activation == "leaky_relu") {
			nodeGradsM[z] = output.unaryExpr(leakyReluGrad);
			output = output.unaryExpr(leakyRelu);
		}
		else if (activation == "linear") {
			nodeGradsM[z] = output.unaryExpr([](double x) {return 1.0; });
		}
#pragma endregion

		// reshape the output
		for (int f = 0; f < numFilters; f++) {
			for (int i = 0; i < layerOutput[0][0].rows(); i++) {
				for (int j = 0; j < layerOutput[0][0].cols(); j++) {

					double dotP = output(f, i * layerOutput[0][0].cols() + j);

					layerOutput[z][f](i, j) = dotP;
				}
			}
		}
	}

	return Tensor::tensorWrap(layerOutput);
}

Tensor ConvoLayer::backward(const Tensor& dyTensor) {
	
	std::vector<Eigen::MatrixXd> dy = dyTensor.matrix3d;
	
	if (dy[0].rows() != numFilters || dy[0].cols() != outputHeight * outputWidth) {
		std::cout << "dy dims don't match: " << dy[0].rows() << " != " << numFilters << " || " << dy[0].cols() << " != " << outputHeight * outputWidth << " at layer: " << name << std::endl;
	}
	
	// Apply activation gradient
	for (int z = 0; z < batchSize; z++) {
		dy[z].array() *= nodeGradsM[z].array();
	}

	// calculate w and input gradients 
	for (int z = 0; z < batchSize; z++) {
		WGrad += dy[z] * XMat[z].transpose();
		XGrad[z] = W.transpose() * dy[z]; // get w.transpose() out of the loop
	}
	
	//calculate B gradients 
	for (int z = 0; z < batchSize; z++) {
		BGradients += dy[z].rowwise().sum(); // myb wrong?
	}

	//turn XGrad into a 4d tensor
	std::vector<std::vector<Eigen::MatrixXd>> XGrad4d(batchSize, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd::Zero(inputHeight, inputWidth)));
	for (int z = 0; z < batchSize; z++) {
		for (int i = 0; i < layerOutput[0][0].rows(); i++) {
			for (int j = 0; j < layerOutput[0][0].cols(); j++) {
				int ii = i * strides.first;
				int jj = j * strides.second;

				for (int c = 0; c < inputChannels; c++) {
					//Eigen::Map<Eigen::VectorXd> v1(x[z][c].block(ii, jj, kernelSize, kernelSize).data(), kernelSize * kernelSize);
					//XMat[z].block(c * kernelSize * kernelSize, i * layerOutput[0][0].cols() + j, kernelSize * kernelSize, 1) = v1;


					Eigen::Map<Eigen::VectorXd> v1(XGrad[z].block(c * kernelSize * kernelSize, i * layerOutput[0][0].cols() + j, kernelSize * kernelSize, 1).data(), kernelSize*kernelSize);
					XGrad4d[z][c].block(ii, jj, kernelSize, kernelSize) = Eigen::MatrixXd::Map(v1.data(), kernelSize, kernelSize);
				}
			}
		}
	}
	//reshape Xgrad4d into a 3d tensor
	std::vector<Eigen::MatrixXd> outputGradientsM = std::vector<Eigen::MatrixXd>(batchSize, Eigen::MatrixXd(inputChannels, inputHeight * inputWidth));
	for (int z = 0; z < batchSize; z++) {
		int matrixSize = inputHeight * inputWidth;
		for (int f = 0; f < inputChannels; f++) {
			Eigen::RowVectorXd channel = Eigen::RowVectorXd::Map(XGrad4d[z][f].data(), XGrad4d[z][f].size());
			outputGradientsM[z].row(f) = channel;
		}
	}


	//return XGrad
	return Tensor::tensorWrap(outputGradientsM);
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
			WGrad += W * lambda / batchSize;
		}
	}

	int inputChannels = x[0].size();

	//Gradient clipping
	double maxNorm = 1.0;
	double norm = WGrad.norm();
	if (norm > maxNorm) {
		WGrad = WGrad * (maxNorm / norm);
	}


	vdw = beta1 * vdw + (1 - beta1) * WGrad;
	vdb = beta1 * vdb + (1 - beta1) * BGradients;

	sdw = beta2 * sdw + (1 - beta2) * WGrad.cwiseProduct(WGrad);
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
	WGrad.setZero();
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
	int inputChannels = x[0].size();

	// adding the dw
	for (int i = 0; i < WGrad.rows(); i++) {
		for (int j = 0; j < WGrad.cols(); j++) {
			dO.push_back(WGrad(i, j));
		}
	}
	
	//adding the db
	for (int i = 0; i < BGradients.size(); i++) {
		dO.push_back(BGradients(i));
	}
}