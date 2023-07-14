#include <vector>
#include <Eigen/Dense>
#include <algorithm>
#include <utility>

#ifndef LAYERS_CPP
#define LAYERS_CPP
#include "Layers.cpp"
#endif 

class ConvoLayer : public Layers {
private:
	int numFilters, kernelSize, padding, batchSize;
	std::pair<int, int> strides;
	std::string activation;

	std::vector<std::vector<Eigen::MatrixXd>> W, WGradients; // (numFilters, channels, h, w)
	std::vector<std::vector<Eigen::MatrixXd>> layerOutput, outputGradients, nodeGrads; //(batch_size, numFilters, outputHeight, outputWidth)
	Eigen::VectorXd b, BGradients; //(numFilters)
	std::vector<std::vector<Eigen::MatrixXd>> x; // (batch_size, channels, h, w)
public:
	ConvoLayer(int numFilters, int kernelSize, std::pair<int, int> strides, int padding, std::string activation) : numFilters(numFilters), kernelSize(kernelSize), strides(strides), activation(activation), padding(padding) {
		b = Eigen::VectorXd(numFilters);
		BGradients = Eigen::VectorXd(numFilters);
	}

	void initSizes(int batchSize1, int inputChannels, int inputHeight, int inputWidth) {
		batchSize = batchSize1;
		int outputHeight = (inputHeight - kernelSize + 2 * padding) / strides.first + 1;
		int outputWidth = (inputWidth - kernelSize + 2 * padding) / strides.second + 1;
		W = std::vector<std::vector<Eigen::MatrixXd>>(numFilters, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd::Random(kernelSize, kernelSize)));
		WGradients = std::vector<std::vector<Eigen::MatrixXd>>(numFilters, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd::Zero(kernelSize, kernelSize)));
		layerOutput = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(numFilters, Eigen::MatrixXd(outputHeight, outputWidth)));
		nodeGrads = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(numFilters, Eigen::MatrixXd(outputHeight, outputWidth)));
		x = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd(inputHeight + 2 * padding, inputWidth + 2 * padding)));
		outputGradients = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd(inputHeight, inputWidth)));
	}

	std::vector<std::vector<Eigen::MatrixXd>> forward(std::vector<std::vector<Eigen::MatrixXd>> input) {
		// add the padding
		for (int z = 0; z < batchSize; z++) {
			for (int c = 0; c < input[0].size(); c++) {
				x[z][c].setZero();  // Set the entire matrix to zero
				x[z][c].block(padding, padding, input[0][0].rows(), input[0][0].cols()) = input[z][c];
			}
		}


		for (int z = 0; z < batchSize; z++) {
			for (int f = 0; f < W.size(); f++) {
				for (int i = 0; i < layerOutput[0][0].rows(); i++) {
					for (int j = 0; j < layerOutput[0][0].cols(); j++) {
						int ii = i * strides.first;
						int jj = j * strides.second;

						double dotP = 0.0;
						for (int c = 0; c < x[0].size(); c++) {
							Eigen::Map<Eigen::VectorXd> v1(W[f][c].data(), W[f][c].size());
							Eigen::Map<Eigen::VectorXd> v2(x[z][c].block(ii, jj, kernelSize, kernelSize).data(), kernelSize * kernelSize);
							dotP += v1.dot(v2);
						}

						dotP += b[f];
						// apply activation function (relu in this case)
						nodeGrads[z][f](i, j) = dotP > 0 ? 1 : 0;
						if (activation == "relu") dotP = std::max(0.0, dotP);
						layerOutput[z][f](i, j) = dotP;
					}
				}
			}
		}

		// Calculate the nodeGrad


		return layerOutput;
	}

	std::vector<std::vector<Eigen::MatrixXd>> backward(std::vector<std::vector<Eigen::MatrixXd>> dy) {

		// Apply activation gradient
		for (int z = 0; z < dy.size(); z++) {
			for (int f = 0; f < dy[0].size(); f++) {
				dy[z][f] = dy[z][f].array() * nodeGrads[z][f].array();
			}
		}


		// Calculate WGradient
		for (int z = 0; z < batchSize; z++) {

			for (int f = 0; f < W.size(); f++) {
				for (int c = 0; c < W[0].size(); c++) {

					for (int i = 0; i < kernelSize; i++) {
						for (int j = 0; j < kernelSize; j++) {

							Eigen::Map<Eigen::VectorXd> v1(dy[z][f].data(), dy[z][f].size());
							Eigen::Map<Eigen::VectorXd> v2(x[z][c].block(i, j, kernelSize, kernelSize).data(), kernelSize * kernelSize);
							WGradients[f][c](i, j) += v1.dot(v2);

						}
					}
				}

				//  Calculate BGradient
				BGradients[f] += dy[z][f].sum();
			}
		}


		// Calculate output gradient
		for (int z = 0; z < batchSize; z++) {
			for (int c = 0; c < W[0].size(); c++) {

				for (int f = 0; f < W.size(); f++) {

					for (int i = 0; i < layerOutput[0][0].rows(); i++) {
						for (int j = 0; j < layerOutput[0][0].cols(); j++) {
							int ii = i * strides.first;
							int jj = j * strides.second;

							for (int iw = 0; iw < kernelSize; iw++) {
								for (int jw = 0; jw < kernelSize; jw++) {
									outputGradients[z][c](ii + iw, jj + jw) += dy[z][c](i, j) * W[f][c](iw, jw);
								}
							}
						}
					}
				}
			}
		}

		return outputGradients;
	}
};
