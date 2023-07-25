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
//#include "./CsvLoader.cpp"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "NNModel.h"
#include "Layers.h"
#include "DenseLayer.h"
#include "ConvoLayer.h"
#include "MaxPoolLayer.h"
#include "FlattenLayer.h"

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

	std::vector<Layers*> input = {
		&ConvoLayer(64, 3, pair(1,1), 1, "relu"),
		&MaxPoolLayer(3,1),
		&ConvoLayer(64, 3, pair(1,1), 1, "relu"),
		&MaxPoolLayer(3,1),
		&ConvoLayer(32, 3, pair(1,1), 1, "relu"),
		&MaxPoolLayer(3,1),
		&FlattenLayer(),
		&DenseLayer(256,"relu"),
		&DenseLayer(82, "softmax")
	};

	NNModel model(input);

	//model.compile(256, xTrain[0].size());

	//model.fit(xTrain, yTrain, 10, 1);

	////Calculate the accuracy
	//std::cout << "Accuracy: " << model.calcAccuracy(xTest, yTest, 0.5) << std::endl;

	// Stop the clock
	auto end = std::chrono::high_resolution_clock::now();

	// Calculate the duration
	std::chrono::duration<double> duration = end - start;

	// Print the runtime
	std::cout << "Runtime: " << duration.count() << " seconds" << std::endl;


	// <------------------------------------------------------------------------------------->

}