#include "Tensor.h"
#include <vector>
#include <iostream>
#include <Eigen/Dense>

Tensor Tensor::tensorWrap(std::vector<double> input) {
    Tensor newTensor;
    newTensor.vector1d = input;
    return newTensor;
}

Tensor Tensor::tensorWrap(Eigen::MatrixXd input) {
    Tensor newTensor;
    newTensor.matrix = input;
    return newTensor;
}

Tensor Tensor::tensorWrap(std::vector<Eigen::MatrixXd> input) {
    Tensor newTensor;
    newTensor.matrix3d = input;
    return newTensor;
}

Tensor Tensor::tensorWrap(std::vector<std::vector<Eigen::MatrixXd>> input) {
    Tensor newTensor;
    newTensor.matrix4d = input;
    return newTensor;
}

Tensor Tensor::tensorWrap(std::vector<std::vector<double>> input) {
    Tensor newTensor;
    newTensor.vector2d = input;
    return newTensor;
}

Tensor Tensor::tensorWrap(double input) {
    Tensor newTensor;
    newTensor.scalar = input;
    return newTensor;
}

void Tensor::print() {
    if (matrix.size() != 0) {
        for (int i = 0; i < matrix.rows(); i++) {
            for (int j = 0; j < matrix.cols(); j++) {
				std::cout << matrix(i, j) << " ";
			}
			std::cout << std::endl << std::endl;
		}
        std::cout << std::endl << std::endl;

    }
    else if (matrix3d.size() != 0) {
        for (int z = 0; z < matrix3d.size(); z++) {
            for (int i = 0; i < matrix3d[0].rows(); i++) {
                for (int j = 0; j < matrix3d[0].cols(); j++) {
                    std::cout<< matrix3d[z](i, j) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl << std::endl;
        }
        std::cout << std::endl;

    }
    else if (matrix4d.size() != 0) {
        for (int b = 0; b < matrix4d.size(); b++) {
            for (int z = 0; z < matrix4d[0].size(); z++) {
                for (int i = 0; i < matrix4d[0][0].rows(); i++) {
                    for (int j = 0; j < matrix4d[0][0].cols(); j++) {
                        std::cout << matrix4d[b][z](i, j) << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}