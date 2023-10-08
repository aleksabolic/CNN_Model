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
#include <unordered_map>

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
#include "CsvLoader.h"


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
	input.push_back(std::make_shared<DenseLayer>(5, "leaky_relu"));
	input.push_back(std::make_shared<DenseLayer>(5, "leaky_relu"));
	input.push_back(std::make_shared<DenseLayer>(3, "linear"));

	/*std::string xTrainPath = "C:\\Users\\aleks\\OneDrive\\Desktop\\Logic Regression\\x_train.csv";
	std::string yTrainPath = "C:\\Users\\aleks\\OneDrive\\Desktop\\Logic Regression\\y_train.csv";
	std::vector<std::vector<double>> xTrain;
	std::vector<int> yTrain;
	CsvLoader::LoadX(xTrain, xTrainPath);
	CsvLoader::LoadY(yTrain, yTrainPath);*/


	std::string xTrainPath = "C:\\Users\\aleks\\OneDrive\\Desktop\\iris_ds\\x_iris.csv";
	std::string yTrainPath = "C:\\Users\\aleks\\OneDrive\\Desktop\\iris_ds\\y_iris.csv";

	std::vector<std::vector<double>> xTrain;
	std::vector<std::string> yTrainRaw;

	CsvLoader::LoadX(xTrain, xTrainPath);
	CsvLoader::LoadY(yTrainRaw, yTrainPath);

	std::vector<int> yTrain(yTrainRaw.size());
	std::unordered_map<std::string, int> uniqueVals;
	
	//Encode string labels to integers
	int index = 1;
	for (int i = 0; i < yTrainRaw.size(); i++) {
		if (!uniqueVals[yTrainRaw[i]]) {
			uniqueVals[yTrainRaw[i]] = index;
			yTrain[i] = index-1;
			index++;
		}
		else {
			yTrain[i] = uniqueVals[yTrainRaw[i]]-1;
		}
	}

	int batchSize = 20;

	DataLoader<std::vector<std::vector<double>>, std::vector<int>> dataLoader = DataLoader<std::vector<std::vector<double>>, std::vector<int>>(xTrain, yTrain, batchSize);
	dataLoader.ShuffleData();

	NNModel model(input);

	//Loss* loss = new BinaryCrossEntropy();
	Loss* loss = new SparseCategoricalCrossEntropy();
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
	model.fit(dataLoader, 50, 0.001,false);

	std::cout<<"Accuracy: "<<model.calcAccuracy(dataLoader, false)<<std::endl;

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