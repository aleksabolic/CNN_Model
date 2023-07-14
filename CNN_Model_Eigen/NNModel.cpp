#include <vector>
#include <Eigen/Dense>

#ifndef DENSELAYER_CPP
#define DENSELAYER_CPP
#include "./DenseLayer.cpp"
#endif // DENSELAYER_CPP

#ifndef CONVOLAYER_CPP
#define CONVOLAYER_CPP
#include "./ConvoLayer.cpp"
#endif // CONVOLAYER_CPP

#ifndef IMAGELOADER.CPP
#define IMAGELOADER.CPP
#include "./ImageLoader.cpp"
#endif

#include "./Loss.cpp"
#include <algorithm>

class NNModel {
private:

	int batchSize = -1;

	void propagateInput(Eigen::MatrixXd x) {
		layers[0].forward(x);
		for (int i = 1; i < layers.size(); i++) {
			layers[i].forward(layers[i - 1].layerOutput);
		}
	}

	void propagateInput(vector<vector<Eigen::MatrixXd>> x) {
		layers[0].forward(x);
		for (int i = 1; i < layers.size(); i++) {
			layers[i].forward(layers[i - 1].layerOutput);
		}
	}

	void propagateGradient(Eigen::MatrixXd dy) {
		layers[layers.size() - 1].backward(dy);
		for (int i = layers.size() - 2; i >= 0; i--) {
			layers[i].backward(layers[i + 1].outputGradients);
		}
	}

	void propagateSize(int batchSize, int inputSize, int index = 0) {
		if (index == layers.size()) return;
		layers[index].initSizes(inputSize, batchSize);
		propagateSize(layers[index].layerOutput.rows(), layers[index].layerOutput.cols(), index + 1); // mistake myb?
	}

	Eigen::MatrixXd calcCostGradient(Eigen::MatrixXd yHat, std::vector<double> y) {

		Eigen::MatrixXd gradients = Eigen::MatrixXd(yHat.rows(), 1);
		for (int i = 0; i < y.size(); i++) {
			double x = yHat(i, 0);
			if ((y[i] == 1 && x == 0) || (y[i] == 0 && x == 1)) {
				gradients(i, 0) = 10000;
				continue;
			}
			if (y[i] == 1) {
				gradients(i, 0) = -1.0 / x;
				continue;
			}
			gradients(i, 0) = 1.0 / (1.0 - x);
		}
		return gradients;
	}

	void adamOptimizer(double alpha, double T, double e = 10e-7, double beta1 = 0.9, double beta2 = 0.999) {

		//init s and v for both w and b
		std::vector<Eigen::MatrixXd> sw = std::vector<Eigen::MatrixXd>(layers.size());
		std::vector<Eigen::MatrixXd> vw = std::vector<Eigen::MatrixXd>(layers.size());
		std::vector<Eigen::RowVectorXd> sb = std::vector<Eigen::RowVectorXd>(layers.size());
		std::vector<Eigen::RowVectorXd> vb = std::vector<Eigen::RowVectorXd>(layers.size());

		for (int l = 0; l < layers.size(); l++) {
			sw[l] = Eigen::MatrixXd::Zero(layers[l].w.rows(), layers[l].w.cols());
			vw[l] = Eigen::MatrixXd::Zero(layers[l].w.rows(), layers[l].w.cols());
			sb[l] = Eigen::RowVectorXd::Zero(layers[l].b.size());
			vb[l] = Eigen::RowVectorXd::Zero(layers[l].b.size());
		}

		for (int t = 1; t < T; t++) { // or check for convergence
			for (int l = 0; l < layers.size(); l++) {

				auto square = [](double x) {return x * x; };
				auto root = [](double x) {return sqrt(x); };
				auto addE = [](double x) {return x + 10e-7; };

				// update s and v for w
				vw[l] = beta1 * vw[l] + (1 - beta1) * layers[l].WGradients;
				sw[l] = beta2 * sw[l] + (1 - beta2) * layers[l].WGradients.unaryExpr(square);

				Eigen::MatrixXd vCorr = vw[l] / (1 - pow(beta1, t));
				Eigen::MatrixXd sCorr = sw[l] / (1 - pow(beta2, t));

				Eigen::MatrixXd rootS = sCorr.unaryExpr(root);

				layers[l].w = layers[l].w - alpha * (vCorr.cwiseQuotient(rootS.unaryExpr(addE)));


				// update s and v for b
				vb[l] = beta1 * vb[l] + (1 - beta1) * layers[l].BGradients;
				sb[l] = beta2 * sb[l] + (1 - beta2) * layers[l].BGradients.unaryExpr(square);

				Eigen::RowVectorXd vbCorr = vb[l] / (1 - pow(beta1, t));
				Eigen::RowVectorXd sbCorr = sb[l] / (1 - pow(beta2, t));

				Eigen::RowVectorXd rS = sbCorr.unaryExpr(root);

				layers[l].b = layers[l].b - alpha * (vbCorr.cwiseQuotient(rS.unaryExpr(addE)));
			}
		}
	}

public:

	std::vector<DenseLayer> layers;

	NNModel(const std::vector<DenseLayer>& layers) : layers(layers) {

	}

	double calcCost(Eigen::MatrixXd x, std::vector<double> y) {
		double cost = 0.0;

		propagateInput(x);

		Eigen::MatrixXd yHat = layers[layers.size() - 1].layerOutput;

		for (int i = 0; i < y.size(); i++) {

			double loss = Loss::binaryCrossEntropy(yHat(i, 0), y[i]);
			cost += loss;
		}
		return cost / y.size();
	}


	void compile(int batchSize1, int inputSize) {
		batchSize = batchSize1;

		propagateSize(batchSize, inputSize);
	}


	void fit(std::vector<std::vector<double>> input, std::vector<double> y, int epochs, double alpha, bool shuffle = false) {


		Eigen::MatrixXd x(input.size(), input[0].size());

		// Fill the Eigen::MatrixXd with the values from the std::vector
		for (size_t i = 0; i < input.size(); ++i) {
			for (size_t j = 0; j < input[0].size(); ++j) {
				x(i, j) = input[i][j];
			}
		}

		for (int j = 0; j < epochs; j++) {

			//if (shuffle) {
			//	//Shuffle the x
			//	std::random_device rd;
			//	std::mt19937 rng(rd());

			//	// Shuffle the vector
			//	std::shuffle(x.begin(), x.end(), rng);
			//}
			std::vector<double> batchY = std::vector<double>(batchSize);

			for (int i = 0; i < y.size(); i++) {


				if ((i % batchSize == 0 || i == y.size() - 1) && i) {

					//handle x and batchY index missmatch
					if (i == y.size() - 1) {
						std::vector<double> batchTemp;
						for (int j = i - batchSize + 1; j < y.size(); j++) {
							batchTemp.push_back(y[j]);
						}
						batchY = batchTemp;
					}

					if (i == y.size() - 1 && i % batchSize != 0) {
						propagateInput(x.middleRows(i - batchSize + 1, batchSize));
					}
					else {
						propagateInput(x.middleRows(i - batchSize, batchSize));
					}

					Eigen::MatrixXd yHat = layers[layers.size() - 1].layerOutput;

					Eigen::MatrixXd dy = calcCostGradient(yHat, batchY); // Binary Cross Entropy Loss

					propagateGradient(dy);


					adamOptimizer(0.0001, 15);

					/*for (DenseLayer& layer : layers) {
						layer.gradientDescent(alpha);

					}*/
				}
				batchY[i % batchSize] = y[i];

			}

			std::cout << "Epoch: " << j << " Cost: " << calcCost(x, y) << std::endl;
		}
		std::cout << "Finished Training " << std::endl;

	}

	void softmaxGradient(Eigen::MatrixXd yHat, std::vector<double> yTrue) {

	}

	


	void train(vector<vector<Eigen::MatrixXd>> dataSet, vector<std::string> dataLabels) {
		propagateInput(dataSet);
		Eigen::MatrixXd yHat = layers[layers.size() - 1].layerOutput;

		Eigen::MatrixXd dy = softmaxGradient(yHat, dataLabels); // Softmax gradient
		propagateGradient(dy);
		adamOptimizer(0.0001, 15);
	}

	void fit(std::string path, int epochs) {

		for (int w = 0; w < epochs; w++) {
			ImageLoader::readImages(path, batchSize, [this](vector<vector<Eigen::MatrixXd>> dataSet, vector<std::string> dataLabels) {
				this->train(dataSet, dataLabels);
			});
		}
	}


	Eigen::MatrixXd predict(Eigen::MatrixXd x) {
		propagateInput(x);
		return layers[layers.size() - 1].layerOutput;
	}

	double calcAccuracy(std::vector<std::vector<double>> input, std::vector<double> y, double delimiter) {

		Eigen::MatrixXd x(input.size(), input[0].size());

		// Fill the Eigen::MatrixXd with the values from the std::vector
		for (size_t i = 0; i < input.size(); ++i) {
			for (size_t j = 0; j < input[0].size(); ++j) {
				x(i, j) = input[i][j];
			}
		}

		// Calculate the accuracy
		Eigen::MatrixXd rawPredict = predict(x);
		std::vector<double> yPred;

		for (int i = 0; i < rawPredict.rows(); i++) {
			rawPredict(i, 0) >= delimiter ? yPred.push_back(1) : yPred.push_back(0);
		}

		double absSum = 0;
		int numLabels = y.size();
		for (int i = 0; i < numLabels; i++) {
			absSum += abs(yPred[i] - y[i]);
		}
		absSum /= numLabels;
		return 100 * (1 - absSum);
	}
};