//#include "Tensor.h"
//#include <vector>
//#include <iostream>
//#include <Eigen/Dense>
//
//template <typename T>
//Tensor::Tensor(T input) {
//    setTypeAndData(input);
//}
//
//Tensor::~Tensor() {
//    cleanup();
//}
//
//void Tensor::cleanup() {
//    switch (type) {
//    case TensorType::Vector1D:
//        data.vector1d.~vector();
//        break;
//    case TensorType::Matrix:
//        data.matrix.~Matrix();
//        break;
//    case TensorType::Matrix3D:
//		data.matrix3d.~vector();
//		break;
//    case TensorType::Matrix4D:
//        data.matrix4d.~vector();
//        break;
//    case TensorType::Vector2D:
//        data.vector2d.~vector();
//		break;
//    case TensorType::CvMat:
//        data.cvMat.~Mat();
//        break;
//    case TensorType::CvMat4D:
//		data.cvMat4d.~vector();
//		break;
//    default:
//        break;
//    }   
//}
//
//template <typename T>
//void Tensor::setTypeAndData(T input) {
//    if constexpr (std::is_same_v<T, std::vector<double>>) {
//        type = TensorType::Vector1D;
//        new (&data.vector1d) std::vector<double>(input);
//    }
//    else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
//        type = TensorType::Matrix;
//        new (&data.matrix) Eigen::MatrixXd(input);
//    }
//    else if constexpr (std::is_same_v<T, std::vector<Eigen::MatrixXd>) {
//        type = TensorType::Matrix3D;
//        new (&data.matrix3d) std::vector<Eigen::MatrixXd>(input);
//    }
//    else if constexpr (std::is_same_v<T, std::vector<std::vector<Eigen::MatrixXd>>>) {
//		type = TensorType::Matrix4D;
//		new (&data.matrix4d) std::vector<std::vector<Eigen::MatrixXd>>(input);
//	}
//    else if constexpr (std::is_same_v<T, std::vector<std::vector<double>>>) {
//		type = TensorType::Vector2D;
//		new (&data.vector2d) std::vector<std::vector<double>>(input);
//	}
//    else if constexpr (std::is_same_v<T, cv::Mat>) {
//        type = TensorType::CvMat;
//        new (&data.cvMat) cv::Mat(input);
//    }
//    else if constexpr (std::is_same_v<T, std::vector<cv::Mat>>) {
//		type = TensorType::CvMat4D;
//		new (&data.cvMat4d) std::vector<cv::Mat>(input);
//	}
//    else {
//        type = TensorType::Scalar;
//        data.scalar = static_cast<double>(input);
//    }
//}
//
//Tensor Tensor::operator+(Tensor other) {
//
//}
//
//void Tensor::print() {
//    /*if (matrix.size() != 0) {
//        for (int i = 0; i < matrix.rows(); i++) {
//            for (int j = 0; j < matrix.cols(); j++) {
//				std::cout << matrix(i, j) << " ";
//			}
//			std::cout << std::endl << std::endl;
//		}
//        std::cout << std::endl << std::endl;
//
//    }
//    else if (matrix3d.size() != 0) {
//        for (int z = 0; z < matrix3d.size(); z++) {
//            for (int i = 0; i < matrix3d[0].rows(); i++) {
//                for (int j = 0; j < matrix3d[0].cols(); j++) {
//                    std::cout<< matrix3d[z](i, j) << " ";
//                }
//                std::cout << std::endl;
//            }
//            std::cout << std::endl << std::endl;
//        }
//        std::cout << std::endl;
//
//    }
//    else if (matrix4d.size() != 0) {
//        for (int b = 0; b < matrix4d.size(); b++) {
//            for (int z = 0; z < matrix4d[0].size(); z++) {
//                for (int i = 0; i < matrix4d[0][0].rows(); i++) {
//                    for (int j = 0; j < matrix4d[0][0].cols(); j++) {
//                        std::cout << matrix4d[b][z](i, j) << " ";
//                    }
//                    std::cout << std::endl;
//                }
//                std::cout << std::endl;
//            }
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
//    }*/
//}

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
                    std::cout << matrix3d[z](i, j) << " ";
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