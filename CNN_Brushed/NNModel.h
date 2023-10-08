#pragma once

#include <vector>
#include <Eigen/Dense>
#include <unordered_map>

#include "./Layers/DenseLayer.h"
#include "./Layers/ConvoLayer.h"
#include "./Layers/FlattenLayer.h"
#include "./Layers/MaxPoolLayer.h"


#include "ImageLoader.h"
#include "DataLoader.h"

#include "Loss.h"
#include <algorithm>

class NNModel {
private:

	int batchSize = -1;

	double datasetSize = 0;

	Loss* loss_ptr;

	std::unordered_map<std::string, int> classNames;

	Tensor propagateInput(const Tensor& x);

	void propagateGradient(const Tensor& dy);

	void propagateSize(const std::unordered_map<std::string, int>& sizes);

	Eigen::MatrixXd softmax(Eigen::MatrixXd x);

public:

	std::vector<std::shared_ptr<Layers>> layers;

	double modelAccuracy = 0;

	NNModel(const std::vector<std::shared_ptr<Layers>>& layersInput);

	// compilation for 1d inputs 
	void compile(int batchSize1, int inputSize, Loss* loss_pointer);

	// compilation for 2d inputs (images)
	void compile(int batchSize1, int inputChannels, int inputHeight, int inputWidth, Loss* loss_pointer);

	void fit(std::vector<std::vector<double>> input, std::vector<double> y, int epochs, double alpha, bool shuffle = false);

	void train(std::vector<std::vector<Eigen::MatrixXd>>& dataSet, std::vector<std::string>& dataLabels);

	void fit(std::string path, int epochs, std::vector<std::string> classNamesS);

	//fit method with dataloader class
	template<class typeX, class typeY>
	void fit(DataLoader<typeX, typeY>& dataLoader, int epochs, double alpha, bool isBinary);

	// Use templates maybe?
	Eigen::MatrixXd predict(Eigen::MatrixXd x);

	Eigen::MatrixXd predict(std::vector < std::vector < Eigen::MatrixXd > > x);

	double calcAccuracy(std::vector<std::vector<double>> input, std::vector<int> y, double delimiter);

	void calcAccuracy(std::vector<std::vector<Eigen::MatrixXd>>& dataSet, std::vector<std::string>& dataLabels);

	template<class typeX, class typeY>
	double calcAccuracy(DataLoader<typeX, typeY> dataLoader, bool binaryClassif);

	double accuracy(std::string path, std::vector<std::string> classNamesS);

	void loadWeights(const std::string& filename);

	void saveWeights(const std::string& filename);

	//testing 
	void checkGrad(std::vector<std::vector<Eigen::MatrixXd>>& dataSet, std::vector<std::string>& dataLabels);

	void gradientChecking(std::string path, std::vector<std::string> classNamesS);
	void gradientChecking(std::vector<std::vector<Eigen::MatrixXd>> x, std::vector<int> y);
	//testing
};

//fit method with data loader
template<class typeX, class typeY>
void NNModel::fit(DataLoader<typeX, typeY>& dataLoader, int epochs, double alpha, bool isBinary) {

	Eigen::MatrixXd input(dataLoader.batchSize, dataLoader.inputSize);

	for (int j = 0; j < epochs; j++) {
		dataLoader.LoadData([&](typeX& x, typeY& y) {

			// convert x to eigen matrix
			for (int i = 0; i < x.size(); i++) {
				input.row(i) = Eigen::Map<Eigen::RowVectorXd>(x[i].data(), 1, x[i].size());
			}

			Eigen::MatrixXd yHat = propagateInput(Tensor::tensorWrap(input)).matrix;

			// If its multiclass classification
			if (!isBinary) {
				yHat = softmax(yHat);
			}

			Eigen::MatrixXd dy = loss_ptr->gradient(yHat, y);

			//cout << dy << endl;

			propagateGradient(Tensor::tensorWrap(dy));

			// gradeint descent
			for (auto& layer : layers) {
				layer->gradientDescent(alpha);
			}
			printf("Epoch: %d Cost: %f\n", j, loss_ptr->cost(yHat, y));

		});
	}
}

template<class typeX, class typeY>
double NNModel::calcAccuracy(DataLoader<typeX, typeY> dataLoader, bool binaryClassif) {

	int correct = 0;
	int total = 0;
	Eigen::MatrixXd input(dataLoader.batchSize, dataLoader.inputSize);

	dataLoader.LoadData([&](typeX& x, typeY& y) {
		// convert x to eigen matrix
		for (int i = 0; i < x.size(); i++) {
			input.row(i) = Eigen::Map<Eigen::RowVectorXd>(x[i].data(), 1, x[i].size());
		}
		Eigen::MatrixXd yHat = propagateInput(Tensor::tensorWrap(input)).matrix;

		for (int z = 0; z < yHat.rows(); z++) {
			if (binaryClassif) {
				if (yHat(z, 0) > 0.5 && y[z] == 1) {
					correct++;
				}
				else if (yHat(z, 0) < 0.5 && y[z] == 0) {
					correct++;
				}
			}
			else {
				Eigen::MatrixXd::Index maxIndex;
				yHat.row(z).maxCoeff(&maxIndex);
				if (y[z] == maxIndex) {
					correct++;
				}
			}
			total++;
		}
	});

	return 100 * (double)correct / total;
}