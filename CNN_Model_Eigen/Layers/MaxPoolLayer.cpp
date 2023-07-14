#include <vector>
#include <Eigen/Dense>

#ifndef LAYERS_CPP
#define LAYERS_CPP
#include "Layers.cpp"
#endif 

// I dont need to save the input ??
class MaxPoolLayer : public Layers{
public:
	int kernelSize, batchSize, stride;
	std::vector<std::vector<Eigen::MatrixXd>> layerOutput, outputGradients, gradGate; //(batch_size, numChannels, outputHeight, outputWidth)
	//std::vector<std::vector<Eigen::MatrixXd>> x; // (batch_size, channels, h, w)

	MaxPoolLayer(int kernelSize, int stride) : kernelSize(kernelSize), stride(stride) {}

	void initSizes(int batchSize1, int inputChannels, int inputHeight, int inputWidth) {
		batchSize = batchSize1;
		int outputHeight = (inputHeight - kernelSize) / stride + 1;
		int outputWidth = (inputWidth - kernelSize) / stride + 1;
		layerOutput = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(inputChannels , Eigen::MatrixXd(outputHeight, outputWidth)));
		gradGate = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd::Zero(inputHeight, inputWidth)));
		outputGradients = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd(inputHeight, inputWidth)));
		//x = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd(inputHeight, inputWidth)));
	}

	std::vector<std::vector<Eigen::MatrixXd>> forward(std::vector<std::vector<Eigen::MatrixXd>> input) {
		for (int z = 0; z < batchSize; z++) {
			for (int c = 0; c < input[0].size(); c++) {

				for (int i = 0; i < layerOutput[0][0].rows(); i++) {
					for (int j = 0; j < layerOutput[0][0].cols(); j++) {
						int ii = i * stride;
						int jj = j * stride;

						// find the max pixel in the input channel
						double maxVal = -DBL_MAX;
						int maxII, maxJJ;
						for (int relativeI = 0; relativeI < kernelSize; relativeI++) {
							for (int relativeJ = 0; relativeJ < kernelSize; relativeJ++) {
								if (input[z][c](ii + relativeI, jj + relativeJ) > maxVal) {
									maxVal - input[z][c](ii + relativeI, jj + relativeJ);
									maxII = ii + relativeI;
									maxJJ = jj + relativeJ;
								}
							}
						}
						gradGate[z][c](maxII, maxJJ) = 1;
						layerOutput[z][c](i, j) = maxVal;
					}
				}
			}
		}
		return layerOutput;
	}

	std::vector<std::vector<Eigen::MatrixXd>> backward(std::vector<std::vector<Eigen::MatrixXd>> dy) {
		for (int z = 0; z < dy.size(); z++) {
			for (int c = 0; c < dy[0].size(); c++) {

				for (int i = 0; i < dy[0][0].rows(); i++) {
					for (int j = 0; j < dy[0][0].cols(); j++) {
						int ii = i * stride;
						int jj = j * stride;

						for (int relativeI = 0; relativeI < kernelSize; relativeI++) {
							for (int relativeJ = 0; relativeJ < kernelSize; relativeJ++) {
								gradGate[z][c](ii + relativeI, jj + relativeJ) *= dy[z][c](i, j);
							}
						}
					}
				}
			}
		}
		return gradGate;
	}
};