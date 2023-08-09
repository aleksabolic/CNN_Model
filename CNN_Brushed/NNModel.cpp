#include <vector>
#include <Eigen/Dense>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <fstream>

#include "NNModel.h"

#include "./Layers/DenseLayer.h"
#include "./Layers/ConvoLayer.h"
#include "./Layers/FlattenLayer.h"
#include "./Layers/MaxPoolLayer.h"

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

void NNModel::compile(int batchSize1, int inputSize, Loss* loss_pointer) {
	loss_ptr = loss_pointer;
	batchSize = batchSize1;

	std::unordered_map<std::string, int> sizes;
	sizes["input size"] = inputSize;
	sizes["batch size"] = batchSize;

	propagateSize(sizes);
}

void NNModel::compile(int batchSize1, int inputChannels, int inputHeight, int inputWidth, Loss* loss_pointer) {
	loss_ptr = loss_pointer;
	batchSize = batchSize1;

	std::unordered_map<std::string, int> sizes;
	sizes["input channels"] = inputChannels;
	sizes["input height"] = inputHeight;
	sizes["input width"] = inputWidth;
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

Eigen::MatrixXd NNModel::softmax(Eigen::MatrixXd x) {
	auto softmaxFix = [](double x) {return std::max(x, 1e-7); };
	auto exponent = [](double x) {return exp(x); };

	// subtract the maximum value from each row
	Eigen::VectorXd rowMax = x.rowwise().maxCoeff();
	x.colwise() -= rowMax;

	//hacky fix
	x.unaryExpr(softmaxFix);

	x = x.unaryExpr(exponent);
	for (int z = 0; z < x.rows(); z++) {
		double sum = x.row(z).sum();
		// hack fix
		if (sum == 0) {
			sum = 1;
		}
		x.row(z) /= sum;
	}
	return x;
}

void NNModel::train(std::vector<std::vector<Eigen::MatrixXd>>& dataSet, std::vector<std::string>& dataLabels) {
	//checkGrad(dataSet, dataLabels);//testing
	printf("Started training...\n");

	Eigen::MatrixXd yHat = propagateInput(Tensor::tensorWrap(dataSet)).matrix;

	// Convert the string labels to int labels
	Eigen::VectorXi labels = Eigen::VectorXi::Zero(dataLabels.size());
	for (int i = 0; i < dataLabels.size(); i++) {
		labels[i] = classNames[dataLabels[i]];
	}

	// Apply softmax to logits
	Eigen::MatrixXd softYHat = softmax(yHat);

	Eigen::MatrixXd dy = loss_ptr->gradient(softYHat, labels);

	propagateGradient(Tensor::tensorWrap(dy));

	// gradeint descent
	for (auto& layer : layers) {
		layer->gradientDescent(1);
	}

	printf("Finished training...  Cost: %f\n", loss_ptr->cost(softYHat, labels));


	//saveWeights("./Model/firstModel");
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

		printf("Epoch: %d\n", e);
		saveWeights("./Model/firstModel");
	}
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

				Eigen::MatrixXd dy = loss_ptr->gradient(yHat, batchY); // Binary Cross Entropy Loss

				propagateGradient(Tensor::tensorWrap(dy));

				// gradeint descent
				for (auto& layer : layers) {
					layer->gradientDescent(alpha);
				}
				printf("Epoch: %d Cost: %f\n", j, loss_ptr->cost(yHat, batchY));
			}
			batchY[i % batchSize] = y[i];

		}
	}
	std::cout << "Finished Training " << std::endl;

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
	double absSum = 0;
	// Calculate the accuracy for each batch
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

			std::vector<double> yPred;

			for (int i = 0; i < yHat.rows(); i++) {
				yHat(i, 0) >= delimiter ? yPred.push_back(1) : yPred.push_back(0);
			}

			for (int i = 0; i < yHat.rows(); i++) {
				absSum += std::abs(yPred[i] - batchY[i]);
			}
		}
		batchY[i % batchSize] = y[i];

	}

	absSum /= y.size();
	return 100 * (1 - absSum);
}

void NNModel::calcAccuracy(std::vector<std::vector<Eigen::MatrixXd>>& dataSet, std::vector<std::string>& dataLabels) {
	Eigen::MatrixXd yHat = predict(dataSet);

	std::cout << yHat.row(0).sum() << std::endl;

	// Convert the string labels to int labels
	Eigen::VectorXi labels = Eigen::VectorXi::Zero(dataLabels.size());
	for (int i = 0; i < dataLabels.size(); i++) {
		labels[i] = classNames[dataLabels[i]];
	}

	for (int z = 0; z < yHat.rows(); z++) {
		Eigen::MatrixXd::Index maxIndex;
		yHat.row(z).maxCoeff(&maxIndex);
		std::cout<< "Predicted: " << maxIndex << " Actual: " << labels[z] << std::endl;
		if (labels[z] == maxIndex) {
			modelAccuracy++;
		}
	}

	datasetSize += yHat.rows();
	std::cout<< "Accuracy: " << modelAccuracy << std::endl;
}

double NNModel::accuracy(std::string path, std::vector<std::string> classNamesS) {

	// create the map with the string to its index pairs
	for (int i = 0; i < classNamesS.size(); i++) {
		classNames[classNamesS[i]] = i;
	}

	modelAccuracy = 0;
	datasetSize = 0;

	ImageLoader::readImages(path, batchSize, [this](std::vector<std::vector<Eigen::MatrixXd>>& dataSet, std::vector<std::string>& dataLabels) {
		this->calcAccuracy(dataSet, dataLabels);
		});

	
	modelAccuracy /= datasetSize;
	return 100 * (1- modelAccuracy);
}



void NNModel::checkGrad(std::vector<std::vector<Eigen::MatrixXd>>& dataSet, std::vector<std::string>& dataLabels) {
	
	//Convert the string labels to int labels
	Eigen::VectorXi labels = Eigen::VectorXi::Zero(dataLabels.size());
	for (int i = 0; i < dataLabels.size(); i++) {
		labels[i] = classNames[dataLabels[i]];
	}


	//<--------------------------------------------------------------->

	Eigen::MatrixXd yHat = propagateInput(Tensor::tensorWrap(dataSet)).matrix;

	Eigen::MatrixXd dy = loss_ptr->gradient(softmax(yHat), labels);

	propagateGradient(Tensor::tensorWrap(dy));

	std::vector<int> indexes;
	std::vector<double> dO;
	int lastSize = 0;
	for (auto& layer : layers) {
		/*std::cout<<"============================"<<std::endl;
		std::cout<<layer->W<<std::endl;
		std::cout << "============================" << std::endl;*/

		layer->addStuff(dO);
		if (lastSize != dO.size()) {
			indexes.push_back(dO.size());
			lastSize = dO.size();
		}
	}
	//scale the dO by batch size
	for (int i = 0; i < dO.size(); i++) {
		dO[i] /= batchSize;
	}
	std::cout << dO.size() << std::endl;
	// Calculate dOapprox

	std::vector<double> dOapprox;
	double epsilon = 1e-9;
	for (auto& layer : layers) {
		for (int i = 0; i < layer->W.rows(); i++) {
			for (int j = 0; j < layer->W.cols(); j++) {
				double temp = layer->W(i, j);
				layer->W(i, j) = temp + epsilon;
				Eigen::MatrixXd yHatPlus = propagateInput(Tensor::tensorWrap(dataSet)).matrix;
				double costPlus = loss_ptr->cost(softmax(yHatPlus), labels);
				layer->W(i, j) = temp - epsilon;
				Eigen::MatrixXd yHatMinus = propagateInput(Tensor::tensorWrap(dataSet)).matrix;
				double costMinus = loss_ptr->cost(softmax(yHatMinus), labels);
				layer->W(i, j) = temp;
				dOapprox.push_back((costPlus - costMinus) / (2 * epsilon));
			}
		}
		for (int i = 0; i < layer->b.size(); i++) {
			double temp = layer->b[i];
			layer->b[i] = temp + epsilon;
			Eigen::MatrixXd yHatPlus = propagateInput(Tensor::tensorWrap(dataSet)).matrix;
			double costPlus = loss_ptr->cost(softmax(yHatPlus), labels);
			layer->b[i] = temp - epsilon;
			Eigen::MatrixXd yHatMinus = propagateInput(Tensor::tensorWrap(dataSet)).matrix;
			double costMinus = loss_ptr->cost(softmax(yHatMinus), labels);
			layer->b[i] = temp;
			dOapprox.push_back((costPlus - costMinus) / (2 * epsilon));
		}
	}	

	double normDiff = 0.0;
	double normDO = 0.0;
	double normDOApprox = 0.0;

	for (int i = 0; i < dO.size(); i++) {
		normDiff += (dO[i] - dOapprox[i]) * (dO[i] - dOapprox[i]);
		normDO += dO[i] * dO[i];
		normDOApprox += dOapprox[i] * dOapprox[i];
	}

	normDiff = std::sqrt(normDiff);
	normDO = std::sqrt(normDO);
	normDOApprox = std::sqrt(normDOApprox);

	double sumNorm = normDO + normDOApprox;
	double relativeDifference = normDiff / sumNorm;

	std::cout << "Relative difference: " << relativeDifference << std::endl;
	
	int index = 0;
	int count = 0;
	if (relativeDifference > 1e-4 || true) {
		for (int i = 0; i < dO.size(); i++) {

			if (i == indexes[index]) {
				std::cout <<"----------------------------------------" << std::endl;
				index++;

			}	
			std::cout << dO[i] << " " << dOapprox[i] << "           Diff: " << dO[i] - dOapprox[i] << "        Percentage: "<< (dO[i] / dOapprox[i]) <<std::endl;

		}
		std::cin.get();
		/*Tensor A = Tensor::tensorWrap(dataSet);
		for (int i = 0; i < layers.size(); i++) {
			A.print();
			A = layers[i]->forward(A);
		}
		A.print();*/
		Eigen::MatrixXd yHat = propagateInput(Tensor::tensorWrap(dataSet)).matrix;

		Eigen::MatrixXd dy = loss_ptr->gradient(softmax(yHat), labels);

		Tensor A = Tensor::tensorWrap(dy);

		for (int i = layers.size() - 1; i >= 0; i--) {
			std::cout<<"============================"<<std::endl;
			A.print();
			std::cout << "============================" << std::endl;

			A = layers[i]->backward(A);
		}
	}
	//saveWeights("./Model/test_model"); 
}

void NNModel::gradientChecking(std::string path, std::vector<std::string> classNamesS) {

	// create the map with the string to its index pairs
	for (int i = 0; i < classNamesS.size(); i++) {
		classNames[classNamesS[i]] = i;
	}

	ImageLoader::readImages(path, batchSize, [this](std::vector<std::vector<Eigen::MatrixXd>>& dataSet, std::vector<std::string>& dataLabels) {
		this->checkGrad(dataSet, dataLabels);
		});
	
}

void NNModel::gradientChecking(std::vector<std::vector<double>> input, std::vector<double> y) {
	//convert x to matrix
	Eigen::MatrixXd x(input.size(), input[0].size());

	// Fill the Eigen::MatrixXd with the values from the std::vector
	for (size_t i = 0; i < input.size(); ++i) {
		for (size_t j = 0; j < input[0].size(); ++j) {
			x(i, j) = input[i][j];
		}
	}

	Eigen::MatrixXd yHat = propagateInput(Tensor::tensorWrap(x)).matrix;

	Eigen::MatrixXd dy = loss_ptr->gradient(yHat, y);

	propagateGradient(Tensor::tensorWrap(dy));

	std::vector<double> dO;
	for (auto& layer : layers) {
		layer->addStuff(dO);
	}
	//scale the dO by batch size
	for (int i = 0; i < dO.size(); i++) {
		dO[i] /= batchSize;
	}
	std::cout << dO.size() << std::endl;
	// Calculate dOapprox
	std::vector<double> dOapprox;
	double epsilon = 1e-4;
	for (auto& layer : layers) {
		for (int i = 0; i < layer->W.rows(); i++) {
			for (int j = 0; j < layer->W.cols(); j++) {
				double temp = layer->W(i, j);
				layer->W(i, j) = temp + epsilon;
				Eigen::MatrixXd yHatPlus = propagateInput(Tensor::tensorWrap(x)).matrix;
				double costPlus = loss_ptr->cost(yHatPlus, y);
				layer->W(i, j) = temp - epsilon;
				Eigen::MatrixXd yHatMinus = propagateInput(Tensor::tensorWrap(x)).matrix;
				double costMinus = loss_ptr->cost(yHatMinus, y);
				layer->W(i, j) = temp;
				dOapprox.push_back((costPlus - costMinus) / (2 * epsilon));
			}
		}
		for (int i = 0; i < layer->b.size(); i++) {
			double temp = layer->b[i];
			layer->b[i] = temp + epsilon;
			Eigen::MatrixXd yHatPlus = propagateInput(Tensor::tensorWrap(x)).matrix;
			double costPlus = loss_ptr->cost(yHatPlus, y);
			layer->b[i] = temp - epsilon;
			Eigen::MatrixXd yHatMinus = propagateInput(Tensor::tensorWrap(x)).matrix;
			double costMinus = loss_ptr->cost(yHatMinus, y);
			layer->b[i] = temp;
			dOapprox.push_back((costPlus - costMinus) / (2 * epsilon));
		}
	}

	double normDiff = 0.0;
	double normDO = 0.0;
	double normDOApprox = 0.0;

	for (int i = 0; i < dO.size(); i++) {
		normDiff += (dO[i] - dOapprox[i]) * (dO[i] - dOapprox[i]);
		normDO += dO[i] * dO[i];
		normDOApprox += dOapprox[i] * dOapprox[i];
	}

	normDiff = std::sqrt(normDiff);
	normDO = std::sqrt(normDO);
	normDOApprox = std::sqrt(normDOApprox);

	double sumNorm = normDO + normDOApprox;
	double relativeDifference = normDiff / sumNorm;

	std::cout << "Relative difference: " << relativeDifference << std::endl;
	for (int i = 0; i < dO.size(); i++) {
		std::cout << dO[i] << " "<< dOapprox[i] << std::endl;
	}
}