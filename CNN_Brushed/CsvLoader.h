#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

class CsvLoader {
public:
	static void LoadX(std::vector<std::vector<double>>& x, std::string path) {

		std::ifstream file(path);
		if (file.is_open()) {
			std::string line;
			while (std::getline(file, line)) {
				// Process each line of the CSV file
				std::stringstream ss(line);
				std::string token;
				std::vector<double> val;

				while (std::getline(ss, token, ',')) {
					double value = std::stod(token); // Convert token to double
					val.push_back(value);
				}
				x.push_back(val);
			}
			file.close();
		}
		else {
			std::cout << "Failed to open file!" << std::endl;
		}
	}

	static void LoadY(std::vector<double>& y, std::string path) {

		std::ifstream file(path);
		if (file.is_open()) {
			std::string line;
			while (std::getline(file, line)) {
				// Process each line of the CSV file
				double value = std::stod(line); // Convert token to double

				y.push_back(value);
			}
			file.close();
		}
		else {
			std::cout << "Failed to open file!" << std::endl;
		}
	}

	static void LoadY(std::vector<int>& y, std::string path) {

		std::ifstream file(path);
		if (file.is_open()) {
			std::string line;
			while (std::getline(file, line)) {
				// Process each line of the CSV file
				double value = std::stoi(line); // Convert token to double

				y.push_back(value);
			}
			file.close();
		}
		else {
			std::cout << "Failed to open file!" << std::endl;
		}
	}

	static void LoadY(std::vector<std::string>& y, std::string path) {
		std::ifstream file(path);
		if (file.is_open()) {
			std::string line;
			while (std::getline(file, line)) {
				// Process each line of the CSV file
				y.push_back(line);
			}
			file.close();
		}
		else {
			std::cout << "Failed to open file!" << std::endl;
		}
	}
};