#pragma once
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
namespace fs = filesystem;

class ImageLoader {
public:

	static vector<string> subfoldersNames(string directory);

	static void readImages(string directory, int batchSize, std::function<void(std::vector<std::vector<Eigen::MatrixXd>>&, std::vector<std::string>&) > callback);

	static Mat convertEigenToCv(const std::vector<Eigen::MatrixXd>& eigenImages);

	static void meanImage(std::string directory, std::string save_dir, int height, int width);
};