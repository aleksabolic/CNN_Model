#include <Eigen/Dense>
#include <vector>
#include "./Layers.cpp"

class FlattenLayer{
public:
	int batchSize, inputChannels, inputHeight, inputWidth;
	void initSizes(int batchSize1, int inputChannels1, int inputHeight1, int inputWidth1) {
		batchSize = batchSize1;
		inputChannels = inputChannels1;
		inputHeight = inputHeight1;
		inputWidth = inputWidth1;
	}

	Eigen::MatrixXd forward(std::vector<std::vector<Eigen::MatrixXd>> input) {

		Eigen::MatrixXd output = Eigen::MatrixXd(batchSize, input[0].size() * input[0][0].size());

		for (int z = 0; z < input.size(); z++) {
			int index = 0;
			for (int c = 0; c < input[0].size(); c++) {
				Eigen::VectorXd vec = Eigen::Map<Eigen::VectorXd>(input[z][c].data(), input[z][c].size());
				output.row(z).segment(index, vec.size()) = vec;
				index += vec.size();
			}
		}

		return output;
	}
	
	std::vector<std::vector<Eigen::MatrixXd>> backward(Eigen::MatrixXd dy) {

		std::vector<std::vector<Eigen::MatrixXd>> output = std::vector<std::vector<Eigen::MatrixXd>>(batchSize, std::vector<Eigen::MatrixXd>(inputChannels, Eigen::MatrixXd(inputHeight, inputWidth)));
	
		for (int z = 0; z < dy.size(); z++) {
			int matrixSize = inputHeight * inputWidth;
			for (int c = 0; c < inputChannels; c++) {
				Eigen::VectorXd channel = dy.row(z).segment(c * matrixSize, matrixSize);
				output[z][c] = Eigen::Map<const Eigen::MatrixXd>(channel.data(), inputHeight, inputWidth);
			}
		}
		
		return output;
	}
};