#include <math.h>

class Sigmoid {
public:

	static double sigmoid(double x) {
		return 1.0 / (1.0 + exp(-1 * x));
	}

	static double sigmoidGradient(double x) {
		return sigmoid(x) * (1 - sigmoid(x));
	}
};