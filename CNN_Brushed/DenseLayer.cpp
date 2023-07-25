#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <unordered_map>

#include "DenseLayer.h"
#include "Sigmoid.h"

//Constructor
DenseLayer::DenseLayer(int numNodes, const std::string& activation) : activation(activation), numNodes(numNodes) {
	BGradients = Eigen::RowVectorXd::Zero(numNodes);
	b = Eigen::RowVectorXd::Random(numNodes);
}

std::unordered_map<std::string, int> DenseLayer::initSizes(std::unordered_map<std::string, int> sizes) {
	int inputSize = sizes["input size"];
	batchSize = sizes["batch size"];

	w = Eigen::MatrixXd::Random(inputSize, numNodes); // Not a mistake this is w transpose
	WGradients = Eigen::MatrixXd::Zero(inputSize, numNodes);
	layerOutput = Eigen::MatrixXd(batchSize, numNodes);
	x = Eigen::MatrixXd(batchSize, inputSize);
	outputGradients = Eigen::MatrixXd(batchSize, inputSize);

	if (activation == "softmax") {
		softmaxNodeGrads = std::vector<Eigen::MatrixXd>(batchSize, Eigen::MatrixXd(numNodes, numNodes));
	}
	else {
		nodeGrads = Eigen::MatrixXd(batchSize, numNodes);
	}

	// output sizes
	std::unordered_map<std::string, int> outputSizes;
	outputSizes["batch size"] = batchSize;
	outputSizes["input size"] = numNodes;
	return outputSizes;
}

void DenseLayer::uploadWeightsBias(std::vector<std::vector<double>> wUpload, std::vector<double> bUpload) {
	if (wUpload.size() != w.rows() || wUpload[0].size() != numNodes) {
		// <--------- Throw error ---------->
		std::cout << wUpload.size() << " " << wUpload[0].size() << " | " << w.rows() << " " << numNodes;
		std::cout << ":(" << std::endl;
	}
	else {
		// Transpose the w 
		for (int i = 0; i < wUpload.size(); i++) {
			for (int j = 0; j < wUpload[0].size(); j++) {
				w(i, j) = wUpload[i][j];
			}
		}
		for (int i = 0; i < bUpload.size(); i++) {
			b(i) = bUpload[i];
		}
	}

}

Tensor DenseLayer::forward(Tensor inputTensor) {

	Eigen::MatrixXd xInput = inputTensor.matrix;
	x = xInput; // batch matrix

	Eigen::MatrixXd wx = x * w;

	wx.rowwise() += b;

	auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
	auto sigmoidDeriv = [](double x) { return Sigmoid::sigmoid(x) * (1 - Sigmoid::sigmoid(x)); };

	auto relu = [](double x) {return x > 0 ? x : 0.0; };
	auto reluGrad = [](double x) {return x > 0 ? 1.0 : 0.0; };

	auto exponent = [](double x) {return exp(x); };

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

		wx = wx.unaryExpr(exponent);
		for (int z = 0; z < wx.rows(); z++) {
			double sum = wx.row(z).sum();
			wx.row(z) /= sum;
		}

		// Calculate softmax node grads
		for (int z = 0; z < batchSize; z++) {
			for (int i = 0; i < numNodes; i++) {
				for (int j = 0; j < numNodes; j++) {
					softmaxNodeGrads[z](i, j) = (i == j) * (wx(z, j) * (1 - wx(z, j))) + (i != j) * (-wx(z, j) * wx(z, i));
				}
			}
		}
	}
	else {
		// Throw an error
		std::cout << "This function is not suported" << std::endl;
	}

	layerOutput = wx;
	return Tensor::tensorWrap(wx);
}

Tensor DenseLayer::backward(Tensor dyTensor) {

	Eigen::MatrixXd dy = dyTensor.matrix;

	// Applying the activation gradient
	if (activation != "softmax") {
		dy = dy.cwiseProduct(nodeGrads);
	}
	else {

		for (int z = 0; z < dy.rows(); z++) {
			dy.row(z) = dy.row(z) * softmaxNodeGrads[z];
		}
	}

	WGradients = x.transpose() * dy;

	BGradients = (Eigen::MatrixXd::Ones(1, dy.rows()) * dy).row(0);

	outputGradients = dy * w.transpose();

	return Tensor::tensorWrap(outputGradients);
}

void DenseLayer::gradientDescent(double alpha) {
	w = w - ((alpha * WGradients) / batchSize);
	b = b - ((alpha * BGradients) / batchSize);

	// Refresh the gradients
	WGradients.setZero();
	BGradients.setZero();
}
