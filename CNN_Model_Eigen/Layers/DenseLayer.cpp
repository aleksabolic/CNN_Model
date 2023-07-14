#include <vector>
#include <Eigen/Dense>
#include <iostream>

#ifndef SIGMOID_CPP
#define SIGMOID_CPP
#include "../Sigmoid.cpp"
#endif // SIGMOID_CPP

#ifndef LAYERS_CPP
#define LAYERS_CPP
#include "Layers.cpp"
#endif 

class DenseLayer : public Layers{
private:
	int numNodes;
	int inputSize;
	int batchSize;
	std::string activation;
public:
	Eigen::MatrixXd x, w, WGradients, layerOutput, outputGradients, nodeGrads; //w is transposed by default
	Eigen::RowVectorXd b, BGradients;

	//Constructor
	DenseLayer(int numNodes, const std::string& activation) : activation(activation), numNodes(numNodes) {
		BGradients = Eigen::RowVectorXd::Zero(numNodes);
		b = Eigen::RowVectorXd::Random(numNodes);
	}

	void initSizes(int inputSize, int batchSize1) {
		batchSize = batchSize1;
		w = Eigen::MatrixXd::Random(inputSize, numNodes); // Not a mistake this is w transpose
		WGradients = Eigen::MatrixXd::Zero(inputSize, numNodes);
		layerOutput = Eigen::MatrixXd(batchSize, numNodes);
		x = Eigen::MatrixXd(batchSize, inputSize);
		nodeGrads = Eigen::MatrixXd(batchSize, numNodes);
		outputGradients = Eigen::MatrixXd(batchSize, inputSize);
	}

	void uploadWeightsBias(std::vector<std::vector<double>> wUpload, std::vector<double> bUpload) {
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

	Eigen::MatrixXd forward(Eigen::MatrixXd xInput) {
		x = xInput; // batch matrix

		Eigen::MatrixXd wx = x * w;

		wx.rowwise() += b;

		auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
		auto sigmoidDeriv = [](double x) { return Sigmoid::sigmoid(x) * (1 - Sigmoid::sigmoid(x)); };

		auto relu = [](double x) {return x > 0 ? x : 0.0; };
		auto reluGrad = [](double x) {return x > 0 ? 1.0 : 0.0; };

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
		else {
			// Throw an error
			std::cout << "This function is not suported" << std::endl;
		}

		layerOutput = wx;
		return wx;
	}

	Eigen::MatrixXd backward(Eigen::MatrixXd dy) {

		// Applying the activation gradient
		dy = dy.cwiseProduct(nodeGrads);

		WGradients = x.transpose() * dy;

		BGradients = (Eigen::MatrixXd::Ones(1, dy.rows()) * dy).row(0);

		outputGradients = dy * w.transpose();

		return outputGradients;
	}

	void gradientDescent(double alpha) {
		w = w - ((alpha * WGradients) / batchSize);
		b = b - ((alpha * BGradients) / batchSize);

		// Refresh the gradients
		WGradients.setZero();
		BGradients.setZero();
	}

};