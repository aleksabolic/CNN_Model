#include "UnflattenLayer.h"

UnflattenLayer::UnflattenLayer(int outputChannels, int outputHeight, int outputWidth) : outputChannels(outputChannels), outputHeight(outputHeight), outputWidth(outputWidth){
	trainable = false;
}

std::unordered_map<std::string, int> UnflattenLayer::initSizes(std::unordered_map<std::string, int>& sizes) {
	inputSize = sizes["input size"];
	batchSize = sizes["batch size"];

	// output sizes
	std::unordered_map<std::string, int> outputSizes;
	outputSizes["batch size"] = batchSize;
	outputSizes["input channels"] = outputChannels;
	outputSizes["input height"] = outputHeight;
	outputSizes["input width"] = outputWidth;

	return outputSizes;
}

Tensor UnflattenLayer::forward(const Tensor& inputTensor) {

	Eigen::MatrixXd input = inputTensor.matrix;

	/*std::cout << "---------------------flatten            input--------------------------------\n";
	Tensor::tensorWrap(input).print();
	std::cout << "---------------------flatten            input--------------------------------\n";*/

	std::vector<std::vector<Eigen::MatrixXd>> output = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(outputChannels, Eigen::MatrixXd(outputHeight, outputWidth)));
	
	#pragma omp parallel for
	for (int z = 0; z < input.rows(); z++) {
		int matrixSize = outputHeight * outputWidth;
		for (int c = 0; c < outputChannels; c++) {
			Eigen::VectorXd channel = input.row(z).segment(c * matrixSize, matrixSize);
			output[z][c] = Eigen::Map<const Eigen::MatrixXd>(channel.data(), outputHeight, outputWidth);
		}
	}
	return Tensor::tensorWrap(output);
}

Tensor UnflattenLayer::backward(const Tensor& dyTensor) {

	std::vector<std::vector<Eigen::MatrixXd>> dy = dyTensor.matrix4d;


	/*std::cout << "---------------------flatt--------------------------------\n";
	Tensor::tensorWrap(dy).print();
	std::cout << "---------------------flatt--------------------------------\n";*/

	Eigen::MatrixXd output = Eigen::MatrixXd(batchSize, inputSize);

#pragma omp parallel for
	for (int z = 0; z < dy.size(); z++) {
		int index = 0;
		for (int c = 0; c < dy[0].size(); c++) {
			Eigen::VectorXd vec = Eigen::Map<Eigen::VectorXd>(dy[z][c].data(), dy[z][c].size());
			output.row(z).segment(index, vec.size()) = vec;
			index += vec.size();
		}
	}
	return Tensor::tensorWrap(output);
}

void UnflattenLayer::gradientDescent(double alpha) {
	// Nothing to do here
}

void UnflattenLayer::saveWeights(const std::string& filename) {
	// Nothing to do here
}

void UnflattenLayer::loadWeights(const std::string& filename) {
	// Nothing to do here
}

//testing
void UnflattenLayer::addStuff(std::vector<double>& dO) {
	// Nothing to do here
}
//testing