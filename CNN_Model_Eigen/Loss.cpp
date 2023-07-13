#include <math.h>
class Loss {
public:
	static double binaryCrossEntropy(double yHat, double y) {
		if ((yHat == 1 && y == 1) || (yHat == 0 && y == 0)) return 0;
		if ((yHat == 1 && y == 0) || (yHat == 0 && y == 1)) return 30;
		return -y * log(yHat) - (1.0 - y) * log(1.0 - yHat);
	}

};