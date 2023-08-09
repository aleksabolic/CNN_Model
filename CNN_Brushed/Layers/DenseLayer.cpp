#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <random>

#include "DenseLayer.h"
#include "../Sigmoid.h"

//Constructor
DenseLayer::DenseLayer(int numNodes, const std::string& activation, bool regularization) : activation(activation), numNodes(numNodes), regularization(regularization) {
	b = Eigen::VectorXd(numNodes);
	vdb = Eigen::VectorXd::Zero(numNodes);
	sdb = Eigen::VectorXd::Zero(numNodes);
	t = 1;
	trainable = true;
}

std::unordered_map<std::string, int> DenseLayer::initSizes(std::unordered_map<std::string, int>& sizes) {
	int inputSize = sizes["input size"];
	batchSize = sizes["batch size"];

	BGradients = Eigen::VectorXd::Zero(numNodes);
	W = Eigen::MatrixXd(inputSize, numNodes); 
	vdw = Eigen::MatrixXd::Zero(inputSize, numNodes);
	sdw = Eigen::MatrixXd::Zero(inputSize, numNodes);
	WGradients = Eigen::MatrixXd::Zero(inputSize, numNodes);
	layerOutput = Eigen::MatrixXd(batchSize, numNodes);
	x = Eigen::MatrixXd(batchSize, inputSize);
	outputGradients = Eigen::MatrixXd::Zero(batchSize, inputSize);

	if (activation == "softmax") {
		softmaxNodeGrads = std::vector<Eigen::MatrixXd>(batchSize, Eigen::MatrixXd(numNodes, numNodes));
	}
	else {
		nodeGrads = Eigen::MatrixXd(batchSize, numNodes);
	}

	std::random_device rd{};
	std::mt19937 gen{rd()};
	double std_dev = sqrt(2.0 / inputSize);  
	std::normal_distribution<> d{0, std_dev}; // Mean 0, standard deviation calculated by He initialization

	for (int i = 0; i < W.rows(); ++i) {
		for (int j = 0; j < W.cols(); ++j) {
			W(i, j) = d(gen);
		}
	}

	for (int i = 0; i < b.size(); ++i) {
		b(i) = d(gen);
	}


	// output sizes
	std::unordered_map<std::string, int> outputSizes;
	outputSizes["batch size"] = batchSize;
	outputSizes["input size"] = numNodes;

	return outputSizes;
}

void DenseLayer::uploadWeightsBias(std::vector<std::vector<double>> wUpload, std::vector<double> bUpload) {
	if (wUpload.size() != W.rows() || wUpload[0].size() != numNodes) {
		// <--------- Throw error ---------->
		std::cout << wUpload.size() << " " << wUpload[0].size() << " | " << W.rows() << " " << numNodes;
		std::cout << ":(" << std::endl;
	}
	else {
		// Transpose the w 
		for (int i = 0; i < wUpload.size(); i++) {
			for (int j = 0; j < wUpload[0].size(); j++) {
				W(i, j) = wUpload[i][j];
			}
		}
		for (int i = 0; i < bUpload.size(); i++) {
			b(i) = bUpload[i];
		}
	}

}

Tensor DenseLayer::forward(const Tensor& inputTensor) {
	Eigen::MatrixXd xInput = inputTensor.matrix;

	/*std::cout << "---------------------dense            input--------------------------------\n";
	Tensor::tensorWrap(x).print();
	std::cout << "---------------------dense            input--------------------------------\n";*/
	x = xInput;

	Eigen::MatrixXd wx = x * W;

	//change b to row vector
	Eigen::RowVectorXd rowVector = b.transpose();

	wx.rowwise() += rowVector;

	auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
	auto sigmoidDeriv = [](double x) { return Sigmoid::sigmoid(x) * (1 - Sigmoid::sigmoid(x)); };

	auto relu = [](double x) {return x > 0 ? x : 0.0; };
	auto reluGrad = [](double x) {return x > 0 ? 1.0 : 0.0; };

	auto exponent = [](double x) {return exp(x); };

	auto softmaxFix = [](double x) {return std::max(x, 1e-9); };

	//Apply activation function

	if (activation == "relu") {
		nodeGrads = wx.unaryExpr(reluGrad);
		wx = wx.unaryExpr(relu);
	}
	else if (activation == "sigmoid") {
		//Calculate node grads
		nodeGrads = wx.unaryExpr(sigmoidDeriv); 
		wx = wx.unaryExpr(sigmoid);
	}
	else if (activation == "linear") {
		// Nothing
	}
	else if (activation == "softmax") {

		// subtract the maximum value from each row
		Eigen::VectorXd rowMax = wx.rowwise().maxCoeff();
		wx.colwise() -= rowMax;

		//hacky fix
		wx.unaryExpr(softmaxFix);

		wx = wx.unaryExpr(exponent);
		for (int z = 0; z < wx.rows(); z++) {
			double sum = wx.row(z).sum();
			// hack fix
			if (sum == 0) {
				sum = 1;
			}
			wx.row(z) /= sum;
		}


		// Calculate softmax node grads
		/*#pragma omp parallel for
		for (int z = 0; z < batchSize; z++) {
			for (int i = 0; i < numNodes; i++) {
				for (int j = 0; j < numNodes; j++) {
					softmaxNodeGrads[z](i, j) = (i == j) * (wx(z, j) * (1 - wx(z, j))) + (i != j) * (-wx(z, j) * wx(z, i));
				}
			}
		}*/
	}
	else {
		// Throw an error
		std::cout << "This function is not suported" << std::endl;
	}

	layerOutput = wx;
	return Tensor::tensorWrap(wx);
}

Tensor DenseLayer::backward(const Tensor& dyTensor) {

	Eigen::MatrixXd dy = dyTensor.matrix;


	/*std::cout << "---------------------dense--------------------------------\n";
	Tensor::tensorWrap(dy).print();
	std::cout << "---------------------dense--------------------------------\n";*/

	// Applying the activation gradient
	if (activation == "linear") {
		// Do nothing
	}
	else if (activation != "softmax") {
		dy = dy.cwiseProduct(nodeGrads);
	}
	else {

		for (int z = 0; z < dy.rows(); z++) {
			dy.row(z) = dy.row(z) * softmaxNodeGrads[z];
		}
	}
	WGradients = x.transpose() * dy;

	BGradients = (Eigen::MatrixXd::Ones(1, dy.rows()) * dy).row(0);

	outputGradients = dy * W.transpose();


	return Tensor::tensorWrap(outputGradients);
}

// Gradient descent with momentum or adam
void DenseLayer::gradientDescent(double alpha) {

	double beta1 = 0.9;
	double beta2 = 0.999;
	double epsilon = 1e-8;

	//check if the layer should be regularized
	/*if (regularization) {
		std::string regularization = "l2";
		double lambda = 0.01;
		if (regularization == "l2") {
			WGradients += (lambda * W) / batchSize;
		}
	}*/
	
	vdw = beta1 * vdw + (1 - beta1) * WGradients;
	vdb = beta1 * vdb + (1 - beta1) * BGradients;

	sdw = beta2 * sdw + (1 - beta2) * WGradients.cwiseProduct(WGradients);
	sdb = beta2 * sdb + (1 - beta2) * BGradients.cwiseProduct(BGradients);

	//bias correction
	Eigen::MatrixXd vdwCorr = vdw / (1 - pow(beta1, t));
	Eigen::VectorXd vdbCorr = vdb / (1 - pow(beta1, t));

	Eigen::MatrixXd sdwCorr = sdw / (1 - pow(beta2, t));
	Eigen::VectorXd sdbCorr = sdb / (1 - pow(beta2, t));

	W = W - (alpha * vdwCorr).cwiseQuotient(sdwCorr.unaryExpr([](double x) { return sqrt(x) + 1e-8; }));
	b = b - (alpha * vdbCorr).cwiseQuotient(sdbCorr.unaryExpr([](double x) { return sqrt(x) + 1e-8; }));

	//W = W - (alpha * WGradients)/batchSize;
	//b = b - (alpha * BGradients)/batchSize;

	// Refresh the gradients
	WGradients.setZero();
	BGradients.setZero();

	t++;
}

void DenseLayer::saveWeights(const std::string& filename) {
	std::ofstream outfile(filename, std::ios::binary);

	int rows = W.rows();
	int cols = W.cols();
	outfile.write((char*)&rows, sizeof(int));
	outfile.write((char*)&cols, sizeof(int));
	outfile.write((char*)W.data(), rows * cols * sizeof(double));

	int size = b.size();
	outfile.write((char*)&size, sizeof(int));
	outfile.write((char*)b.data(), size * sizeof(double));

	outfile.close();
}

void DenseLayer::loadWeights(const std::string& filename) {
	std::ifstream infile(filename, std::ios::binary);

	int rows, cols;
	infile.read((char*)&rows, sizeof(int));
	infile.read((char*)&cols, sizeof(int));
	infile.read((char*)W.data(), rows * cols * sizeof(double));

	int size;
	infile.read((char*)&size, sizeof(int));
	infile.read((char*)b.data(), size * sizeof(double));

	infile.close();
}

void DenseLayer::addStuff(std::vector<double>& dO) {
	// adding the dw
	for (int i = 0; i < WGradients.rows(); i++) {
		for (int j = 0; j < WGradients.cols(); j++) {
			dO.push_back(WGradients(i, j));
		}
	}

	//adding the db
	for (int i = 0; i < BGradients.size(); i++) {
		dO.push_back(BGradients(i));
	}
}