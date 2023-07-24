#pragma once
#include<vector>
#include<Eigen/Dense>
class Tensor {
public:
	std::vector<double> vector1d;
	Eigen::MatrixXd matrix;
	std::vector<Eigen::MatrixXd> matrix3d;
	std::vector<std::vector<Eigen::MatrixXd>> matrix4d;
	std::vector<std::vector<double>> vector2d;
	double scalar;

	static Tensor tensorWrap(std::vector<double> input);

	static Tensor tensorWrap(Eigen::MatrixXd input);

	static Tensor tensorWrap(std::vector<Eigen::MatrixXd> input);

	static Tensor tensorWrap(std::vector<std::vector<Eigen::MatrixXd>> input);

	static Tensor tensorWrap(std::vector<std::vector<double>> input);

	static Tensor tensorWrap(double input);
};