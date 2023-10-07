#include <vector>
#include <string>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <Eigen/Dense>
#include <chrono>
#include <filesystem>
#include "functional"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "Loss.h"
#include "Scce.h"
#include "Bce.h"
#include "NNModel.h"

#include "./Layers/Layers.h"
#include "./Layers/DenseLayer.h"
#include "./Layers/ConvoLayer.h"
#include "./Layers/MaxPoolLayer.h"
#include "./Layers/FlattenLayer.h"
#include "./Layers/UnflattenLayer.h"

#include "DataLoader.h"

class Print {
public:
	static void print(std::vector < std::vector<double>> a) {
		for (auto row : a) {
			for (auto e : row) {
				std::cout << e << " ";
			}
			std::cout << std::endl;
		}
	}

	static void print(std::vector<double> a) {
		for (auto e : a) {
			std::cout << e << " ";
		}
		std::cout << std::endl;
	}

	static void print(std::vector<std::string> a) {
		for (auto e : a) {
			std::cout << e << " ";
		}
		std::cout << std::endl;
	}
};

class CsvLoader {
public:
	static void LoadX(std::vector<std::vector<double>>& x, std::string path) {

		std::ifstream file(path);
		if (file.is_open()) {
			std::string line;
			while (std::getline(file, line)) {
				// Process each line of the CSV file
				std::stringstream ss(line);
				std::string token;
				std::vector<double> val;

				while (std::getline(ss, token, ',')) {
					double value = std::stod(token); // Convert token to double
					val.push_back(value);
				}
				x.push_back(val);
			}
			file.close();
		}
		else {
			std::cout << "Failed to open file!" << std::endl;
		}
	}

	static void LoadY(std::vector<double>& y, std::string path) {

		std::ifstream file(path);
		if (file.is_open()) {
			std::string line;
			while (std::getline(file, line)) {
				// Process each line of the CSV file
				double value = std::stod(line); // Convert token to double

				y.push_back(value);
			}
			file.close();
		}
		else {
			std::cout << "Failed to open file!" << std::endl;
		}
	}
};

#include <iostream>
#include <functional>

int main() {
	omp_set_num_threads(10);

	auto start = std::chrono::high_resolution_clock::now();

	std::vector<std::shared_ptr<Layers>> input;
	/*input.push_back(std::make_shared<ConvoLayer>(64, 3, pair(1, 1), 0, "leaky_relu"));
	input.push_back(std::make_shared<MaxPoolLayer>(2, 2,0));
	input.push_back(std::make_shared<ConvoLayer>(64, 3, pair(1, 1), 0, "leaky_relu"));
	input.push_back(std::make_shared<MaxPoolLayer>(2, 2,0));
	input.push_back(std::make_shared<ConvoLayer>(32, 3, pair(1, 1), 0, "leaky_relu"));
	input.push_back(std::make_shared<MaxPoolLayer>(2, 2,0));
	input.push_back(std::make_shared<FlattenLayer>());
	input.push_back(std::make_shared<DenseLayer>(256, "relu"));
	input.push_back(std::make_shared<DenseLayer>(82, "linear"));*/
	input.push_back(std::make_shared<DenseLayer>(5, "relu"));
	input.push_back(std::make_shared<DenseLayer>(10, "relu"));
	input.push_back(std::make_shared<DenseLayer>(1, "sigmoid"));


	std::string xTrainPath = "C:\\Users\\aleks\\OneDrive\\Desktop\\Logic Regression\\x_train.csv";
	std::string yTrainPath = "C:\\Users\\aleks\\OneDrive\\Desktop\\Logic Regression\\y_train.csv";

	std::vector<std::vector<double>> xTrain;
	std::vector<double> yTrain;

	CsvLoader::LoadX(xTrain, xTrainPath);
	CsvLoader::LoadY(yTrain, yTrainPath);

	int batchSize = 256;
	/*vector<vector<double>> x(10, vector<double>(10));
	Eigen::MatrixXd mat(10, 10);


	for (int i = 0; i < 10; i++) {
		mat.row(i) = Eigen::Map<Eigen::RowVectorXd>(x[i].data(), 1, x[i].size());
	}*/

	DataLoader<std::vector<std::vector<double>>, std::vector<double>> dataLoader = DataLoader<std::vector<std::vector<double>>, std::vector<double>>(xTrain, yTrain, batchSize);

	//CsvLoader::LoadX(xTrain, "C:\\Users\\aleks\\Downloads\\iris.csv");

	NNModel model(input);

	Loss* loss = new BinaryCrossEntropy();
	//Loss* loss = new SparseCategoricalCrossEntropy();
	//model.compile(32, 1, 45, 45, loss);
	model.compile(batchSize, xTrain[0].size(), loss);

	//model.loadWeights("./Model/scv");
	
	//int batchSize = 256;
	/*int numNodes = 10;
	Eigen::MatrixXd wx(batchSize, numNodes);
	Eigen::VectorXd b(numNodes);

	for (int i = 0; i < numNodes; i++) {
		wx.block(0, i, batchSize, 1).array() += b(i);
	}*/

	////make dataset for grad check
	//std::vector<std::vector<double>> xTrainGrad;
	//std::vector<double> yTrainGrad;
	//for (int i = 0; i < 256; i++) {
	//	xTrainGrad.push_back(xTrain[i]);
	//	yTrainGrad.push_back(yTrain[i]);
	//}

	//model.gradientChecking(xTrainGrad, yTrainGrad);

	//model.fit(xTrain, yTrain, 15, 0.05);
	model.fit(dataLoader, 15, 0.05);

	std::cout<<"Accuracy: "<<model.calcAccuracy(xTrain, yTrain, 0.5)<<std::endl;

	//std::string path = "C:\\Users\\aleks\\OneDrive\\Desktop\\train_images";
	//std::vector<std::string> classNames = ImageLoader::subfoldersNames(path);

	//model.loadWeights("./Model/firstModel");

	//model.gradientChecking(path, classNames);

	//model.fit(path, 2, classNames);

	//Calculate the accuracy
	//std::cout << "Accuracy: " << model.accuracy(path, classNames) << std::endl;

	// Stop the clock
	auto end = std::chrono::high_resolution_clock::now();

	// Calculate the duration
	std::chrono::duration<double> duration = end - start;

	// Print the runtime
	std::cout << "Runtime: " << duration.count() << " seconds" << std::endl;


	// <------------------------------------------------------------------------------------->

}