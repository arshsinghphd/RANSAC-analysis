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

#define N 100
#define N_PARAMS 2
// graph 1 x range
#define X_MIN1 0.0f
#define X_MAX1 0.6f
// graph 2 x range
#define X_MIN2 0.39f
#define X_MAX2 0.99f
// fraction of outliers
#define EPS 0.2

#include "experiments.h"

int main() {
	// initiate two datasets
	float points_x1[N], points_y1[N];
	float points_x2[N], points_y2[N];
	float t1_out, t2_out;

	// n_inliers and n_outliers same for both
	int n_inliers = (int) (1.0f - EPS) * N;
	int n_outliers = N - n_inliers;
	// true params same for both
	float true_params = _TRUE_PARAMS_LINEAR;

	make_data(points_x1, points_y1,
                n_inliers, n_outliers,
                true_params, N_PARAMS,
                NOISE_STD,
                t1_out,
                0,
                NULL, 0.0f);
	make_data(points_x2, points_y2,
                n_inliers, n_outliers,
                true_params, N_PARAMS,
                NOISE_STD,
                t2_out,
                0,
                NULL, 0.0f);

	
	// OLS on each graph
	float ols_params1[N_PARAMS], ols_params2[N_PARAMS];
	fit_model(x1, y1, N1, ols_params1, N_PARAMS);
    fit_model(x2, y2, N2, ols_params2, N_PARAMS);

	// RANSAC on each graph based on experiments.h macros and EPS
    int k = compute_k(EPS, N_PARAMS, FAIL_PROB);
    int d1 = compute_d(EPS, N);
    int d2 = compute_d(EPS, N);
	float ransac_return1[2 + N_PARAMS];
    float ransac_return2[2 + N_PARAMS];
    ransac(x1, y1, N1, N_PARAMS, k, t1, d1, ransac_return1);
    ransac(x2, y2, N2, N_PARAMS, k, t2, d2, ransac_return2);
	

	// create a combined array
    float x_all[N1 + N2], y_all[N1 + N2];
    // fill these
    for (int i = 0; i < N1; i++) { 
    	x_all[i] = x1[i]; 
    	y_all[i] = y1[i]; 
    }
    for (int i = 0; i < N2; i++) { 
    	x_all[N1 + i] = x2[i]; 
    	y_all[N1 + i] = y2[i]; 
    }
    
    // OLS on combined data
    float ols_combined[N_PARAMS];
    fit_model(x_all, y_all, N1 + N2, ols_combined, N_PARAMS);


    // stitch RANSAC models
    float stitched_params[N_PARAMS];
    float threshold = (t1 + t2) / 2.0f;
    stitch_models(x1, y1, N1, ransac_params1,
                  x2, y2, N2, ransac_params2,
                  stitched_params, N_PARAMS, threshold);

    
	return 0
}