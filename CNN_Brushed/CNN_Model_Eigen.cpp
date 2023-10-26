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
	input.push_back(std::make_shared<ConvoLayer>(2, 3, pair(1, 1), 0, "linear", "1."));
	input.push_back(std::make_shared<MaxPoolLayer>(2, 2, 0));
	input.push_back(std::make_shared<ConvoLayer>(2, 3, pair(1, 1), 0, "linear", "2."));
	input.push_back(std::make_shared<MaxPoolLayer>(2, 2, 0));
	input.push_back(std::make_shared<ConvoLayer>(2, 3, pair(1, 1), 0, "linear", "3."));
	input.push_back(std::make_shared<MaxPoolLayer>(2, 2, 0));
	input.push_back(std::make_shared<ConvoLayer>(2, 3, pair(1, 1), 0, "linear", "4."));

	input.push_back(std::make_shared<FlattenLayer>());
	
	/*input.push_back(std::make_shared<ConvoLayer>(2, 10, pair(1, 1), 0, "linear", "3."));
	input.push_back(std::make_shared<FlattenLayer>());*/
	input.push_back(std::make_shared<DenseLayer>(1, "sigmoid"));

	//int batchSize = 1;

	NNModel model(input);

	Loss* loss = new BinaryCrossEntropy();
	model.compile(2, 3, 45, 45, loss);

	std::string path = "C:\\Users\\aleks\\OneDrive\\Desktop\\train_img_bin";
	std::vector<std::string> classNames = ImageLoader::subfoldersNames(path);

	model.gradientChecking(path, classNames);

	// Stop the clock
	auto end = std::chrono::high_resolution_clock::now();

	// Calculate the duration
	std::chrono::duration<double> duration = end - start;

	// Print the runtime
	std::cout << "Runtime: " << duration.count() << " seconds" << std::endl;


	// <------------------------------------------------------------------------------------->

}