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
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "Loss.h"
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

	std::vector<std::shared_ptr<Layers>> input;
	input.push_back(std::make_shared<ConvoLayer>(64,3,pair(1,1),0, "relu"));
	input.push_back(std::make_shared<MaxPoolLayer>(2,2));
	input.push_back(std::make_shared<ConvoLayer>(64, 3, pair(1, 1), 0, "relu"));
	input.push_back(std::make_shared<MaxPoolLayer>(2,2));
	input.push_back(std::make_shared<ConvoLayer>(32, 3, pair(1, 1), 0, "relu"));
	input.push_back(std::make_shared<MaxPoolLayer>(2,2));
	input.push_back(std::make_shared<FlattenLayer>());
	input.push_back(std::make_shared<DenseLayer>(256,"relu"));
	input.push_back(std::make_shared<DenseLayer>(82, "softmax"));

	NNModel model(input);

	model.compile(32, 3, 45, 45);

	std::string path = "C:\\Users\\aleks\\OneDrive\\Desktop\\train_images";
	std::vector<std::string> classNames = ImageLoader::subfoldersNames(path);

	//model.loadWeights("./Model/firstModel");


	model.fit(path, 2, classNames);

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