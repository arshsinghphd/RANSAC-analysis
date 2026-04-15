/* =============================================================================
Motivating Example

I create two overlapping graphs with the same true lienar model. 
I recover the underlying models in two ways:
	1. using fit_model over the appended dataset which is like solving naive 
		OLS, and
	2. using stitch_model which is the RANSAC approach
Later I will plot this data and the models recovered using naive OLS and the 
RANSAC parameters to visually compare the two approaches and demmnstrate the 
goodness of fit.
============================================================================= */

#define N 60
#define N_PARAMS 2
// graph 1 x range
#define X_MIN1 0
#define X_MAX1 59
// graph 2 x range
#define X_MIN2 40
#define X_MAX2 99
// true model params
#define TRUE_A0 0
#define TRUE_A1 1
// noise_std for gaussian noise (mean 0). 
// outliers will lie outside 2 * noise_std band
#define NOISE_STD 0.5
// fraction of outliers
#define EPS 0.2

#include "experiments.h"

int main() {
	return 0
}