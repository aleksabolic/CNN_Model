#include <vector>
#include <Eigen/Dense>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <fstream>

#include "NNModel.h"

#include "DenseLayer.h"
#include "ConvoLayer.h"
#include "FlattenLayer.h"
#include "MaxPoolLayer.h"

#include "ImageLoader.h"

//#include "Loss.h"

NNModel::NNModel(const std::vector<std::shared_ptr<Layers>>& layersInput) : layers(layersInput) {}

Tensor NNModel::propagateInput(const Tensor& x) {
	Tensor A = x;
	for (int i = 0; i < layers.size(); i++) {
		A = layers[i]->forward(A);
	}
	return A;
}

void NNModel::propagateGradient(const Tensor& dy) {
	Tensor A = dy;
	for (int i = layers.size() - 1; i >= 0; i--) {
		A = layers[i]->backward(A);
	}
}

void NNModel::propagateSize(const std::unordered_map<std::string, int>& sizes) {
	auto A = sizes;
	for (auto& layer : layers) {
		A = layer->initSizes(A);
	}
}

void NNModel::compile(int batchSize1, int inputSize) {
	batchSize = batchSize1;

	std::unordered_map<std::string, int> sizes;
	sizes["input size"] = inputSize;
	sizes["batch size"] = batchSize;

	propagateSize(sizes);
}

void NNModel::saveWeights(const std::string& modelName) {

	for (int i = 0; i < layers.size(); i++) {
		if (layers[i]->trainable) {
			std::string file = modelName + std::to_string(i);
			layers[i]->saveWeights(file);
		}
	}
}

void NNModel::loadWeights(const std::string& modelName) {
	
	for (int i = 0; i < layers.size(); i++) {
		if (layers[i]->trainable) {
			std::string file = modelName + std::to_string(i);
			layers[i]->loadWeights(file);
		}
	}
}

void NNModel::compile(int batchSize1, int inputChannels, int inputHeight, int inputWidth) {
	batchSize = batchSize1;

	std::unordered_map<std::string, int> sizes;
	sizes["input channels"] = inputChannels;
	sizes["input height"] = inputHeight;
	sizes["input width"] = inputWidth;
	sizes["batch size"] = batchSize;

	propagateSize(sizes);
}

Eigen::MatrixXd NNModel::calcCostGradient(Eigen::MatrixXd yHat, std::vector<double> y) {

	Eigen::MatrixXd gradients = Eigen::MatrixXd(yHat.rows(), 1);
	for (int i = 0; i < y.size(); i++) {
		double x = yHat(i, 0);
		if ((y[i] == 1 && x == 0) || (y[i] == 0 && x == 1)) {
			gradients(i, 0) = 10000;
			continue;
		}
		if (y[i] == 1) {
			gradients(i, 0) = -1.0 / x;
			continue;
		}
		gradients(i, 0) = 1.0 / (1.0 - x);
	}
	return gradients;
}

//void NNModel::adamOptimizer(double alpha, double T, double e = 10e-7, double beta1 = 0.9, double beta2 = 0.999) {
//
//	//init s and v for both w and b for DenseLayers
//	std::vector<Eigen::MatrixXd> sw = std::vector<Eigen::MatrixXd>(layers.size());
//	std::vector<Eigen::MatrixXd> vw = std::vector<Eigen::MatrixXd>(layers.size());
//	std::vector<Eigen::RowVectorXd> sb = std::vector<Eigen::RowVectorXd>(layers.size());
//	std::vector<Eigen::RowVectorXd> vb = std::vector<Eigen::RowVectorXd>(layers.size());
//
//	for (int l = 0; l < layers.size(); l++) {
//		sw[l] = Eigen::MatrixXd::Zero(layers[l].w.rows(), layers[l].w.cols());
//		vw[l] = Eigen::MatrixXd::Zero(layers[l].w.rows(), layers[l].w.cols());
//		sb[l] = Eigen::RowVectorXd::Zero(layers[l].b.size());
//		vb[l] = Eigen::RowVectorXd::Zero(layers[l].b.size());
//	}
//
//	//init s and v for both w and b for ConvoLayers
//	std::vector< std::vector<std::vector<Eigen::MatrixXd>> > swC =
//
//		auto square = [](double x) {return x * x; };
//	auto root = [](double x) {return sqrt(x); };
//	auto addE = [](double x) {return x + 10e-7; };
//
//	for (int t = 1; t < T; t++) { // or check for convergence
//		for (int l = 0; l < layers.size(); l++) {
//
//			// update s and v for w
//			vw[l] = beta1 * vw[l] + (1 - beta1) * layers[l].WGradients;
//			sw[l] = beta2 * sw[l] + (1 - beta2) * layers[l].WGradients.unaryExpr(square);
//
//			Eigen::MatrixXd vCorr = vw[l] / (1 - pow(beta1, t));
//			Eigen::MatrixXd sCorr = sw[l] / (1 - pow(beta2, t));
//
//			Eigen::MatrixXd rootS = sCorr.unaryExpr(root);
//
//			layers[l].w = layers[l].w - alpha * (vCorr.cwiseQuotient(rootS.unaryExpr(addE)));
//
//
//			// update s and v for b
//			vb[l] = beta1 * vb[l] + (1 - beta1) * layers[l].BGradients;
//			sb[l] = beta2 * sb[l] + (1 - beta2) * layers[l].BGradients.unaryExpr(square);
//
//			Eigen::RowVectorXd vbCorr = vb[l] / (1 - pow(beta1, t));
//			Eigen::RowVectorXd sbCorr = sb[l] / (1 - pow(beta2, t));
//
//			Eigen::RowVectorXd rS = sbCorr.unaryExpr(root);
//
//			layers[l].b = layers[l].b - alpha * (vbCorr.cwiseQuotient(rS.unaryExpr(addE)));
//		}
//	}
//}


double NNModel::calcCost(Eigen::MatrixXd x, std::vector<double> y) {
	double cost = 0.0;

	Eigen::MatrixXd yHat = propagateInput(Tensor::tensorWrap(x)).matrix;

	for (int i = 0; i < y.size(); i++) {

		//double loss = Loss::binaryCrossEntropy(yHat(i, 0), y[i]);
		//cost += loss;
	}
	return cost / y.size();
}

double NNModel::calcCost(std::vector < std::vector < Eigen::MatrixXd > > x, std::vector<std::string> yTrue) {
	double cost = 0.0;

	Eigen::MatrixXd yHat = propagateInput(Tensor::tensorWrap(x)).matrix;

	for (int z = 0; z < yTrue.size(); z++) {
		int yTrueIndex = classNames[yTrue[z]];
		double loss = -log(yHat(z,yTrueIndex));
		cost += loss;
	}
	return cost / yTrue.size();
}

double NNModel::calcBatchCost(Eigen::MatrixXd yHat, std::vector<std::string> yTrue) {
	double cost = 0.0;

	for (int z = 0; z < yTrue.size(); z++) {
		int yTrueIndex = classNames[yTrue[z]];

		// Guard from inf
		if (yHat(z, yTrueIndex) == 0) {
			cost += 30;
		}
		else {
			cost += -log(yHat(z, yTrueIndex));
		}
	}
	return cost / yTrue.size();
}


void NNModel::fit(std::vector<std::vector<double>> input, std::vector<double> y, int epochs, double alpha, bool shuffle) {


	Eigen::MatrixXd x(input.size(), input[0].size());

	// Fill the Eigen::MatrixXd with the values from the std::vector
	for (size_t i = 0; i < input.size(); ++i) {
		for (size_t j = 0; j < input[0].size(); ++j) {
			x(i, j) = input[i][j];
		}
	}

	for (int j = 0; j < epochs; j++) {

		//if (shuffle) {
		//	//Shuffle the x
		//	std::random_device rd;
		//	std::mt19937 rng(rd());

		//	// Shuffle the vector
		//	std::shuffle(x.begin(), x.end(), rng);
		//}
		std::vector<double> batchY = std::vector<double>(batchSize);

		for (int i = 0; i < y.size(); i++) {


			if ((i % batchSize == 0 || i == y.size() - 1) && i) {

				//handle x and batchY index missmatch
				if (i == y.size() - 1) {
					std::vector<double> batchTemp;
					for (int j = i - batchSize + 1; j < y.size(); j++) {
						batchTemp.push_back(y[j]);
					}
					batchY = batchTemp;
				}

				Eigen::MatrixXd yHat;

				if (i == y.size() - 1 && i % batchSize != 0) {
					yHat = propagateInput(Tensor::tensorWrap(x.middleRows(i - batchSize + 1, batchSize))).matrix;
				}
				else {
					yHat = propagateInput(Tensor::tensorWrap(x.middleRows(i - batchSize, batchSize))).matrix;
				}

				Eigen::MatrixXd dy = calcCostGradient(yHat, batchY); // Binary Cross Entropy Loss

				propagateGradient(Tensor::tensorWrap(dy));


				//adamOptimizer(0.0001, 15);

				/*for (DenseLayer& layer : layers) {
					layer.gradientDescent(alpha);

				}*/
			}
			batchY[i % batchSize] = y[i];

		}

		std::cout << "Epoch: " << j << " Cost: " << calcCost(x, y) << std::endl;
	}
	std::cout << "Finished Training " << std::endl;

}

Eigen::MatrixXd NNModel::softmaxGradient(Eigen::MatrixXd yHat, std::vector<std::string> yTrue) {
	Eigen::MatrixXd dy = Eigen::MatrixXd::Zero(yHat.rows(), yHat.cols());

	for (int z = 0; z < dy.rows(); z++) {
		int yTrueIndex = classNames[yTrue[z]];
		if (yHat(z, yTrueIndex) == 0) {
			dy(z, yTrueIndex) = 10000;
		}
		else {
			dy(z, yTrueIndex) = -1.0 / yHat(z, yTrueIndex);
		}
	}
	return dy;
}

void NNModel::train(std::vector<std::vector<Eigen::MatrixXd>> dataSet, std::vector<std::string> dataLabels) {
	std::cout<<"Started training..." << std::endl;

	Eigen::MatrixXd yHat = propagateInput(Tensor::tensorWrap(dataSet)).matrix;

	Eigen::MatrixXd dy = softmaxGradient(yHat, dataLabels); // Softmax gradient
	propagateGradient(Tensor::tensorWrap(dy));

	//adamOptimizer(0.0001, 15);

	// gradeint descent
	for (auto& layer : layers) {
		layer->gradientDescent(0.01);
	}

	std::cout << "Finished training...  Cost: " << calcBatchCost(yHat, dataLabels) <<std::endl;
}

void NNModel::fit(std::string path, int epochs, std::vector<std::string> classNamesS) {

	// create the map for with the string to its index pairs
	for (int i = 0; i < classNamesS.size(); i++) {
		classNames[classNamesS[i]] = i;
	}

	for (int e = 0; e < epochs; e++) {
		ImageLoader::readImages(path, batchSize, [this](std::vector<std::vector<Eigen::MatrixXd>>& dataSet, std::vector<std::string>& dataLabels) {
			this->train(dataSet, dataLabels);
			});

		std::cout << "Epoch: " << e << std::endl;
		saveWeights("./Model/firstModel");
	}
}

Eigen::MatrixXd NNModel::predict(Eigen::MatrixXd x) {
	return propagateInput(Tensor::tensorWrap(x)).matrix;
}

Eigen::MatrixXd NNModel::predict(std::vector < std::vector < Eigen::MatrixXd > > x) {
	return propagateInput(Tensor::tensorWrap(x)).matrix;
}

double NNModel::calcAccuracy(std::vector<std::vector<double>> input, std::vector<double> y, double delimiter) {

	Eigen::MatrixXd x(input.size(), input[0].size());

	// Fill the Eigen::MatrixXd with the values from the std::vector
	for (size_t i = 0; i < input.size(); ++i) {
		for (size_t j = 0; j < input[0].size(); ++j) {
			x(i, j) = input[i][j];
		}
	}

	// Calculate the accuracy
	Eigen::MatrixXd rawPredict = predict(x);
	std::vector<double> yPred;

	for (int i = 0; i < rawPredict.rows(); i++) {
		rawPredict(i, 0) >= delimiter ? yPred.push_back(1) : yPred.push_back(0);
	}

	double absSum = 0;
	int numLabels = y.size();
	for (int i = 0; i < numLabels; i++) {
		absSum += std::abs(yPred[i] - y[i]);
	}
	absSum /= numLabels;
	return 100 * (1 - absSum);
}

double NNModel::calcAccuracy(std::vector < std::vector < Eigen::MatrixXd > > input, std::vector<std::string> yTrue) {

	Eigen::MatrixXd yHat = predict(input);

	double correct = 0;
	for (int z = 0; z < yHat.rows(); z++) {
		Eigen::MatrixXd::Index maxIndex;
		yHat.row(z).maxCoeff(&maxIndex);
		if (classNames[yTrue[z]] == maxIndex) {
			correct++;
		}
	}
	correct /= yHat.rows();
	return 100 * (1- correct);
}
