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
#include "./CsvLoader.cpp"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#ifndef DENSELAYER_CPP
#define DENSELAYER_CPP
#include "./DenseLayer.cpp"
#endif // DENSELAYER_CPP

#ifndef CONVOLAYER_CPP
#define CONVOLAYER_CPP
#include "./ConvoLayer.cpp"
#endif // CONVOLAYER_CPP

#include "./NNModel.cpp"

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

int main() {

	auto start = std::chrono::high_resolution_clock::now();

	std::vector<DenseLayer> input = {
		DenseLayer(5,"relu"),
		DenseLayer(10, "relu"),
		DenseLayer(20, "relu"),
		DenseLayer(1, "sigmoid")
	};
	NNModel model(input);


	//std::string xTrainPath = "C:\\Users\\aleks\\OneDrive\\Desktop\\Logic Regression\\x_train.csv";
	//std::string yTrainPath = "C:\\Users\\aleks\\OneDrive\\Desktop\\Logic Regression\\y_train.csv";

	//std::vector<std::vector<double>> xTrain;
	//std::vector<double> yTrain;


	//CsvLoader::LoadX(xTrain, xTrainPath);
	//CsvLoader::LoadY(yTrain, yTrainPath);

	//std::cout << "X_train size: (" << xTrain.size() << ", " << xTrain[0].size() << ")" << std::endl;
	//std::cout << "Y_train size: " << yTrain.size() << std::endl;

	//model.compile(256, xTrain[0].size());

	//model.fit(xTrain, yTrain, 10, 1);

	//// Loading the test set
	//std::string xTestPath = "C:\\Users\\aleks\\OneDrive\\Desktop\\Logic Regression\\x_test.csv";
	//std::string yTestPath = "C:\\Users\\aleks\\OneDrive\\Desktop\\Logic Regression\\y_test.csv";

	//std::vector<std::vector<double>> xTest;
	//std::vector<double> yTest;


	//CsvLoader::LoadX(xTest, xTestPath);
	//CsvLoader::LoadY(yTest, yTestPath);

	////Calculate the accuracy
	//std::cout << "Accuracy: " << model.calcAccuracy(xTest, yTest, 0.5) << std::endl;

	//// Stop the clock
	//auto end = std::chrono::high_resolution_clock::now();

	//// Calculate the duration
	//std::chrono::duration<double> duration = end - start;

	//// Print the runtime
	//std::cout << "Runtime: " << duration.count() << " seconds" << std::endl;


	// <------------------------------------------------------------------------------------->
	
}