#include <math.h>
#include "Sigmoid.h"

double Sigmoid::sigmoid(double x) {
	return 1.0 / (1.0 + exp(-1 * x));
}

double Sigmoid::sigmoidGradient(double x) {
	return sigmoid(x) * (1 - sigmoid(x));
}