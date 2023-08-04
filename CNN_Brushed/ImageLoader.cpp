#include <string>
#include <filesystem>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "ImageLoader.h"

using namespace std;
using namespace cv;
namespace fs = filesystem;

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

//void ImageLoader::readImages(string directory, int batchSize, std::function<void(std::vector<std::vector<Eigen::MatrixXd>>&, std::vector<std::string>&) > callback) {
//
//	vector<vector<Eigen::MatrixXd>> dataSet;
//	vector<std::string> dataLabels;
//
//	int size = 0;
//	for (const auto& entry : fs::recursive_directory_iterator(directory)) {
//		if (fs::is_regular_file(entry)) {
//			string path = entry.path().string();
//			Mat image = imread(path);
//			if (!image.empty()) {
//				image.convertTo(image, CV_64F); // Convert image to double precision
//				image /= 255.0; // Rescale image
//
//				// Split the image into its color channels
//				vector<Mat> channels(3);
//				split(image, channels);
//
//				vector<Eigen::MatrixXd> image;
//				for (auto& channel : channels) {
//					// Convert cv::Mat to Eigen::Matrix
//					Eigen::MatrixXd eigenImage;
//					cv::cv2eigen(channel, eigenImage);
//
//					// Store the Eigen::Matrix into the vector
//					image.push_back(eigenImage);
//				}
//				dataSet.push_back(image);
//				size++;
//
//				// Extract the parent path and store it
//				string className = entry.path().parent_path().filename().string();
//				dataLabels.push_back(className);
//			}
//			else {
//				cerr << "Failed to open " << path << endl;
//			}
//
//			// <--------check if its the last picture-------->
//			if (size == batchSize) {
//				callback(dataSet, dataLabels);
//
//				// clear the datasets
//				vector<vector<Eigen::MatrixXd>>().swap(dataSet);
//				vector<std::string>().swap(dataLabels);
//				size = 0;
//
//				std::cout<< "Batch loaded" << std::endl;
//			}
//		}
//
//	}
//
//}

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