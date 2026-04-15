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
#define EXAMPLE_CSV "results/exmaple.csv"
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
    float true_params[] = _TRUE_PARAMS_LINEAR;
    make_inliers(x2, y2, n_inliers2, true_params, N_PARAMS, 40.0f, 100.0f);
    add_gaussian_noise(y2, n_inliers2, NOISE_STD);
    add_outliers(x2, y2, n_inliers2, n_outliers2, true_params, N_PARAMS, NOISE_STD);

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
    ransac(x1, y1, N1, N_PARAMS, k, t1_out, d1, ransac_return1);
    ransac(x2, y2, N2, N_PARAMS, k, t2_out, d2, ransac_return2);

    // create a combined array
    float x_all[N + N], y_all[N + N];
    // fill these
    for (int i = 0; i < N; i++) { 
    	x_all[i] = x1[i]; 
    	y_all[i] = y1[i]; 
    }
    for (int i = 0; i < N; i++) { 
    	x_all[N + i] = x2[i]; 
    	y_all[N + i] = y2[i]; 
    }
    
    // OLS on combined data
    float ols_combined[N_PARAMS];
    fit_model(x_all, y_all, N + N, ols_combined, N_PARAMS);


    // stitch RANSAC models
    float stitched_params[N_PARAMS];
    float threshold = (t1_out + t2_out) / 2.0f;
    stitch_models(x1, y1, N, ransac_params1,
                  x2, y2, N, ransac_params2,
                  stitched_params, N_PARAMS, threshold);


    // --- print results ---
    printf("True params: a0=%.4f a1=%.4f\n",
           true_params[0], true_params[1]);
    printf("\nOLS graph 1: 0=%.4f a1=%.4f error=%.4f\n",
           ols_params1[0], ols_params1[1],
           model_error(ols_params1, true_params, N_PARAMS));
    printf("OLS graph 2: a0=%.4f a1=%.4f error=%.4f\n",
           ols_params2[0], ols_params2[1],
           model_error(ols_params2, true_params, N_PARAMS));
    printf("OLS combined: a0=%.4f a1=%.4f error=%.4f\n",
           ols_combined[0], ols_combined[1],
           model_error(ols_combined, true_params, N_PARAMS));
    printf("\nRANSAC graph 1: a0=%.4f a1=%.4f error=%.4f\n",
           ransac_params1[0], ransac_params1[1],
           model_error(ransac_params1, true_params, N_PARAMS));
    printf("RANSAC graph 2: a0=%.4f a1=%.4f error=%.4f\n",
           ransac_params2[0], ransac_params2[1],
           model_error(ransac_params2, true_params, N_PARAMS));
    printf("RANSAC stitched: a0=%.4f a1=%.4f error=%.4f\n",
           stitched_params[0], stitched_params[1],
           model_error(stitched_params, true_params, N_PARAMS));

    // write CSV 
    FILE* fp = fopen(EXAMPLE_CSV, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s\n", EXAMPLE_CSV);
        return 1;
    }
    fprintf(fp, "x,y,graph\n");
    for (int i = 0; i < N; i++)
        fprintf(fp, "%.4f,%.4f,1\n", x1[i], y1[i]);
    for (int i = 0; i < N; i++)
        fprintf(fp, "%.4f,%.4f,2\n", x2[i], y2[i]);
    fclose(fp);

    printf("\nData written to %s\n", EXAMPLE_CSV);
	return 0
}