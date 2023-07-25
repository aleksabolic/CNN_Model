#pragma once
#include <math.h>

class Sigmoid {
public:
	static double sigmoid(double x);

	static double sigmoidGradient(double x);
};