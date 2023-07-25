#include "Tensor.h"
#include <vector>
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

