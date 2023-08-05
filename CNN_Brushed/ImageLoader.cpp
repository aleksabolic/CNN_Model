#include <string>
#include <filesystem>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <random>
#include "ImageLoader.h"

using namespace std;
using namespace cv;
namespace fs = filesystem;

bool areMatricesEqual(const Eigen::MatrixXd& matrix1, const Eigen::MatrixXd& matrix2, const Eigen::MatrixXd& matrix3, double epsilon = 1e-10)
{
	// Ensure matrices have the same size
	if (matrix1.rows() != matrix2.rows() || matrix1.cols() != matrix2.cols() || matrix1.rows() != matrix3.rows() || matrix1.cols() != matrix3.cols())
	{
		std::cout << "Matrices are of different sizes." << std::endl;
		return false;
	}

	// Check if matrices are approximately equal
	if (!matrix1.isApprox(matrix2, epsilon) || !matrix1.isApprox(matrix3, epsilon))
	{
		std::cout << "Not all matrices are equal." << std::endl;
		return false;
	}

	std::cout << "All matrices are equal." << std::endl;
	return true;
}

vector<string> ImageLoader::subfoldersNames(string directory) {
	vector<string> subfolders;

	// Iterate over each entry in the directory
	for (const auto& entry : fs::directory_iterator(directory)) {
		if (fs::is_directory(entry)) { // If the entry is a directory, add it to the subfolders vector
			subfolders.push_back(entry.path().filename().string());
		}
	}
	return subfolders;
}

void ImageLoader::readImages(string directory, int batchSize, std::function<void(std::vector<std::vector<Eigen::MatrixXd>>&, std::vector<std::string>&) > callback) {

	vector<vector<Eigen::MatrixXd>> dataSet;
	vector<std::string> dataLabels;

	// Store all the image paths
	vector<string> allImages;
	for (const auto& entry : fs::recursive_directory_iterator(directory)) {
		if (fs::is_regular_file(entry)) {
			allImages.push_back(entry.path().string());
		}
	}

	// Shuffle the image paths
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::shuffle(allImages.begin(), allImages.end(), std::default_random_engine(seed));

	int size = 0;
	// Loop over shuffled image paths
	for (const auto& path : allImages) {
		Mat image = imread(path);
		if (!image.empty()) {

			image.convertTo(image, CV_64F); // Convert image to double precision
			image /= 255.0; // Rescale image
			
			// Split the image into its color channels
			vector<Mat> channels(3);
			split(image, channels);
			
			vector<Eigen::MatrixXd> image;
			for (auto& channel : channels) {
				// Convert cv::Mat to Eigen::Matrix
				Eigen::MatrixXd eigenImage;
				cv::cv2eigen(channel, eigenImage);
			
				// Store the Eigen::Matrix into the vector
				image.push_back(eigenImage);

				//testing
				break;
				//testing
			}
			
			dataSet.push_back(image);
			size++;

			// Extract the parent path and store it
			fs::path fsPath(path);
			string className = fsPath.parent_path().filename().string();
			dataLabels.push_back(className);

			// <--------check if its the last picture-------->
			if (size == batchSize) {
				callback(dataSet, dataLabels);

				// clear the datasets
				vector<vector<Eigen::MatrixXd>>().swap(dataSet);
				vector<string>().swap(dataLabels);
				size = 0;

				std::cout << "Batch loaded" << std::endl;
			}
		}
		else {
			cerr << "Failed to open " << path << endl;
		}
	}
}