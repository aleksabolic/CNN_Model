//#pragma once
//#include<vector>
//#include<Eigen/Dense>
//#include <opencv2/opencv.hpp>
//
//enum class TensorType {
//	Vector1D,
//	Matrix,
//	Matrix3D,
//	Matrix4D,
//	Vector2D,
//	CvMat,
//	CvMat4D,
//	Scalar
//};
//
//class Tensor {
//private:
//	union Data {
//		std::vector<double> vector1d;
//		Eigen::MatrixXd matrix;
//		std::vector<Eigen::MatrixXd> matrix3d;
//		std::vector<std::vector<Eigen::MatrixXd>> matrix4d;
//		std::vector<std::vector<double>> vector2d;
//		cv::Mat cvMat;
//		std::vector<cv::Mat> cvMat4d;
//
//		double scalar;
//
//		Data() {}
//		~Data() {}
//
//	} data;
//
//	template <typename T>
//	void setTypeAndData(T input);
//
//	void cleanup();
//
//public:
//	TensorType type;
//
//	template <typename T>
//	Tensor(T input);
//
//	~Tensor();
//
//	Tensor operator+(Tensor other);
//
//	Tensor operator-(Tensor other);
//
//	Tensor operator*(Tensor other);
//
//	Tensor operator/(Tensor other);
//
//	Tensor operator+(double other);
//
//
//	void print();
//};

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

	void print();
};