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

void ImageLoader::meanImage(string directory, string save_dir, int height, int width) {
	// Store all the image paths
	vector<string> allImages;
	for (const auto& entry : fs::recursive_directory_iterator(directory)) {
		if (fs::is_regular_file(entry)) {
			allImages.push_back(entry.path().string());
		}
	}

	Eigen::MatrixXd meanImage = Eigen::MatrixXd::Zero(height, width); // Initialize mean image
	Eigen::MatrixXd sumSquared = Eigen::MatrixXd::Zero(height, width); // Initialize sum of squared differences
	int totalImages = 0;

	for (const auto& path : allImages) {
		Mat image = imread(path);
		if (!image.empty()) {
			image.convertTo(image, CV_64F); // Convert image to double precision
			image /= 255.0; // Rescale image
			vector<Mat> channels(3);
			split(image, channels);

			for (auto& channel : channels) {
				Eigen::MatrixXd eigenImage;
				cv::cv2eigen(channel, eigenImage);
				meanImage += eigenImage;
				totalImages++;
				break; // Remove this break if you want to include all channels
			}
		}
		if(totalImages%10000 == 0)
			cout << "Processed " << totalImages << " images." << endl;
	}
	meanImage /= totalImages;

	// Calculate sum of squared differences
	for (const auto& path : allImages) {
		Mat image = imread(path);
		if (!image.empty()) {
			image.convertTo(image, CV_64F);
			image /= 255.0;
			vector<Mat> channels(3);
			split(image, channels);

			for (auto& channel : channels) {
				Eigen::MatrixXd eigenImage;
				cv::cv2eigen(channel, eigenImage);
				sumSquared += (eigenImage - meanImage).array().square().matrix();
				break;
			}
		}
	}
	Eigen::MatrixXd sigmaImage = (sumSquared / totalImages).array().sqrt();

	// Save mean image
	cv::Mat meanCvImage;
	cv::eigen2cv(meanImage, meanCvImage);
	meanCvImage *= 255.0;
	meanCvImage.convertTo(meanCvImage, CV_8U);
	string savePath = save_dir + "/meanImage.png";
	cv::imwrite(savePath, meanCvImage);

	// Save sigma image
	cv::Mat sigmaCvImage;
	cv::eigen2cv(sigmaImage, sigmaCvImage);
	sigmaCvImage *= 255.0;
	sigmaCvImage.convertTo(sigmaCvImage, CV_8U);
	savePath = save_dir + "/sigmaImage.png";
	cv::imwrite(savePath, sigmaCvImage);
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

	//Load the mean image
	Mat meanImage = imread("./MeanImage/meanImage.png");
	meanImage.convertTo(meanImage, CV_64F);
	meanImage /= 255.0;

	//Load the sigma image
	Mat sigmaImage = imread("./MeanImage/sigmaImage.png");
	sigmaImage.convertTo(meanImage, CV_64F);
	sigmaImage /= 255.0;

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

			// Split the mean image into its color channels
			vector<Mat> meanChannels(3);
			split(meanImage, meanChannels);	

			// Split the sigma image into its color channels
			vector<Mat> sigmaChannels(3);
			split(sigmaImage, sigmaChannels);
			
			vector<Eigen::MatrixXd> image;
			for (auto& channel : channels) {
				// Convert cv::Mat to Eigen::Matrix
				Eigen::MatrixXd eigenImage;
				cv::cv2eigen(channel, eigenImage);
			
				// Subtract the mean image and devide with sigma image
				/*Eigen::MatrixXd meanChannel, sigmaChannel;
				cv::cv2eigen(meanChannels[0], meanChannel);
				cv::cv2eigen(sigmaChannels[0], sigmaChannel);
				eigenImage -= meanChannel;*/
				//eigenImage = eigenImage.cwiseQuotient(sigmaChannel);

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

Mat ImageLoader::convertEigenToCv(const std::vector<Eigen::MatrixXd>& eigenImages) {
	if (eigenImages.size() != 3) {
		throw std::invalid_argument("Expected 3 channels in the input vector.");
	}

	// Convert each Eigen::MatrixXd to cv::Mat and rescale
	std::vector<cv::Mat> channels;
	for (const auto& eigenImage : eigenImages) {
		cv::Mat cvImage(eigenImage.rows(), eigenImage.cols(), CV_64F);
		for (int i = 0; i < cvImage.rows; ++i) {
			for (int j = 0; j < cvImage.cols; ++j) {
				cvImage.at<double>(i, j) = eigenImage(i, j) * 255.0;
			}
		}
		channels.push_back(cvImage);
	}

	// Merge channels into a single image
	cv::Mat image;
	cv::merge(channels, image);

	// Convert back to 8-bit image
	image.convertTo(image, CV_8UC3);
	return image;
}