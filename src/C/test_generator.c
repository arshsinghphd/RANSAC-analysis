/**
 * Tests for generator.h functions:
 * make_inliers
 * add_gaussian_noise
 * add_laplace_noise
 * add_structural_bias
 * add_outliers
 * 
 * make_inliers generates inlier points on a polynomial model of degree 
 * n_params - 1, defined by params from lowest to highest degree [a0, a1, ...].
 * 
 * add_outliers appends outlier points guaranteed to lie outside the inlier
 * band defined by the true polynomial model and noise_std, using vertical
 * residual consistent with find_model_inliers.
 */

#include "generator.h"

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

#define ASSERT(test, msg) \
	if (!(test)) { \
		printf("FAIL: %s\n", msg); \
		return 1; \
	}

#define RUN_TEST(test, n) \
    if (test() == 0) \
        printf("PASS test %d: %s\n", n, #test); \

#define N 10
#define X_MIN 0.0
#define X_MAX 9.0
#define N_PARAMS 2


/* ---------- TESTS FOR MAKE_INLIERS ---------- */
/**
 * Tests happy paths for make_inliers with polynomial models of degree 1 (line)
 * and degree 2 (quadratic), and edge cases.
 * 
 * Happy paths:
 * unit slope zero intercept       y = 0 + 1*x
 * negative unit slope             y = (n-1) + (-1)*x
 * float slope zero intercept      y = 0 + 0.5*x
 * unit slope negative intercept   y = (-n+1) + 1*x
 * zero slope zero intercept       y = 0 (flat line)
 * quadratic model                 y = 1 + 1*x + 1*x^2
 * 
 * Edge cases:
 * n < 2               should return -1
 * x_min == x_max      should return -1
 * n_params < 2        should return -1
 */

/** Test happy path: slope 1 intercept 0. */
int test_make_inliers_slope_1_intercept_0(){
	float points_x[N];
	float points_y[N];
	float intercept = 0;
	float slope = 1;
	float params[] = {intercept, slope};
	make_inliers(points_x, points_y, N, params, N_PARAMS, X_MIN, X_MAX);
	for(int i = 0; i < N; i++){
		ASSERT(fabs(points_x[i] - i) < 1e-6, "slope -1 intercept N: points_x wrong");
		ASSERT(fabs(points_y[i] - (intercept + i * slope)) < 1e-6, "slope -1 intercept N: points_y wrong");
	}
	return 0;
}

/** Test happy path: slope -1 intercept N. */
int test_make_inliers_slope_neg1_intercept_n(){
	float points_x[N];
	float points_y[N];
	float intercept = N;
	float slope = -1;
	float params[] = {intercept, slope};
	make_inliers(points_x, points_y, N, params, N_PARAMS, X_MIN, X_MAX);
	for(int i = 0; i < N; i++){
		ASSERT(fabs(points_x[i] - i) < 1e-6, "slope -1 intercept N: points_x wrong");
		ASSERT(fabs(points_y[i] - (intercept + i * slope)) < 1e-6, "slope -1 intercept N: points_y wrong");
	}
	return 0;
}

/** Test happy path: slope 0.5 intercept 0. */
int test_make_inliers_slope_float_intercept_0() {
	float points_x[N];
	float points_y[N];
	float intercept = 0;
	float slope = 0.5;
	float params[] = {intercept, slope};
	make_inliers(points_x, points_y, N, params, N_PARAMS, X_MIN, X_MAX);
	for(int i = 0; i < N; i++){
		ASSERT(fabs(points_x[i] - i) < 1e-6, "slope -1 intercept N: points_x wrong");
		ASSERT(fabs(points_y[i] - (intercept + i * slope)) < 1e-6, "slope -1 intercept N: points_y wrong");
	}
	return 0;
}

/** Test happy path: slope 1 intercept -1. */
int test_make_inliers_slope_1_intercept_neg_float() {
	float points_x[N];
	float points_y[N];
	float intercept = -0.5;
	float slope = 1;
	float params[] = {intercept, slope};
	make_inliers(points_x, points_y, N, params, N_PARAMS, X_MIN, X_MAX);
	for(int i = 0; i < N; i++){
		ASSERT(fabs(points_x[i] - i) < 1e-6, "slope -1 intercept N: points_x wrong");
		ASSERT(fabs(points_y[i] - (intercept + i * slope)) < 1e-6, "slope -1 intercept N: points_y wrong");
	}
	return 0;
}

/** Test happy path: slope 0 intercept 0. */
int test_make_inliers_slope_0_intercept_0() {
	float points_x[N];
	float points_y[N];
	float intercept = 0.0;
	float slope = 0.0;
	float params[] = {intercept, slope};
	make_inliers(points_x, points_y, N, params, N_PARAMS, X_MIN, X_MAX);
	for(int i = 0; i < N; i++){
		ASSERT(fabs(points_x[i] - i) < 1e-6, "slope -1 intercept N: points_x wrong");
		ASSERT(fabs(points_y[i] - (intercept + i * slope)) < 1e-6, "slope -1 intercept N: points_y wrong");
	}
	return 0;
}

/** Test happy path: higher degree (quadratic) model. */
int test_make_inliers_qudratic_model() {
	float points_x[N];
	float points_y[N];
	float a = 1.0;
	float b = 1.0;
	float c = 1.0;
	float params[] = {a, b, c};
	int n_params = 3;
	make_inliers(points_x, points_y, N, params, n_params, X_MIN, X_MAX);
	for(int i = 0; i < N; i++){
		ASSERT(fabs(points_x[i] - i) < 1e-6, "slope -1 intercept N: points_x wrong");
		ASSERT(fabs(points_y[i] - (a + b * i + c * pow(i, 2))) < 1e-6, "slope -1 intercept N: points_y wrong");
	}
	return 0;
}

/** Test edge case: n < 2, should return -1 */
int test_make_inliers_edge_n_lt_2(){
	int n = 1;
	float points_x[n];
	float points_y[n];
	float intercept = 0;
	float slope = 1;
	float params[] = {intercept, slope};
	int ret = make_inliers(points_x, points_y, n, params, N_PARAMS, X_MIN, X_MAX);
	ASSERT(ret == -1, "For n < 2, make_inliers should return -1.");
	return 0;
}

/** Test edge case: x_min == x_max, should return -1 */
int test_make_inliers_edge_xmin_eq_xmax(){
	float points_x[N];
	float points_y[N];
	float intercept = 0;
	float slope = 1;
	float params[] = {intercept, slope};
	float x_min = 10;
	float x_max = 10;
	int ret = make_inliers(points_x, points_y, N, params, N_PARAMS, x_min, x_max);
	ASSERT(ret == -1, "For x_min == x_max, make_inliers should return -1.");
	return 0;
}

/** Test edge case: n_params < 2, should return -1 */
int test_make_inliers_edge_n_params_lt_2(){
	float points_x[N];
	float points_y[N];
	float intercept = 0;
	float params[] = {intercept};
	int n_params = 1;
	int ret = make_inliers(points_x, points_y, N, params, n_params, X_MIN, X_MAX);
	ASSERT(ret == -1, "For n_params < 2, make_inliers should return -1.");
	return 0;
}


/* ------------ TESTS FOR ADD GAUSSIAN NOISE ---------- */
/**
 * Tests happy paths and edge cases for add_gaussian_noise.
 * 
 * Happy paths: all noisy points_y[i] within original points_y[i] +/- 5 * std
 * 	std int 
 * 	std float
 * 
 * Edge cases:
 * n_inliers < 2   should return -1
 * std <= 0        should return -1
 * std == 0        should return -1
 */

/** Test happy path, std integer. */
int test_add_gaussian_noise_std_int() {
	float points_x[N];
	float points_y[N];
	float intercept = 0;
	float slope = 1;
	float params[] = {intercept, slope};
	make_inliers(points_x, points_y, N, params, N_PARAMS, X_MIN, X_MAX);
	float std = 2;
	add_gaussian_noise(points_y, N, std);
	int changed;
	for(int i = 0; i < N; i++){
		changed = 0;
		if(fabs(points_y[i] - (intercept + i * slope)) > 1e-6)
        	changed = 1;
        ASSERT(changed, "std 2: no noise was added");
		ASSERT(fabs(points_y[i] - (intercept + i * slope)) < 3 * std, 
			"std 2: points_y wrong");
	}
	return 0;
}

/** Test happy path, std float. */
int test_add_gaussian_noise_std_float() {
	float points_x[N];
	float points_y[N];
	float intercept = 0;
	float slope = 1;
	float params[] = {intercept, slope};
	make_inliers(points_x, points_y, N, params, N_PARAMS, X_MIN, X_MAX);
	float std = 2.5;
	add_gaussian_noise(points_y, N, std);
	int changed;
	for(int i = 0; i < N; i++){
		changed = 0;
		if(fabs(points_y[i] - (intercept + i * slope)) > 1e-6)
        	changed = 1;
        ASSERT(changed, "std 2.5: no noise was added");
		ASSERT(fabs(points_y[i] - (intercept + i * slope)) < 3 * std, 
			"std 2.5: points_y wrong");
	}
	return 0;
}

/** Test edge case: n_inliers < 2 */
int test_add_gaussian_noise_n_inliers_lt_2() {
	float points_x[N];
	float points_y[N];
	float intercept = 0;
	float slope = 1;
	float params[] = {intercept, slope};
	make_inliers(points_x, points_y, N, params, N_PARAMS, X_MIN, X_MAX);
	int n = 1;	// override for test
	float std = 2.5;
	int ret = add_gaussian_noise(points_y, n, std);
	ASSERT(ret == -1, 
			"with n_inliers 1, add_gaussian_noise should return -1.");
	return 0;
}

/* std == 0 should return -1 */
int test_add_gaussian_noise_std_0() {
	float points_x[N];
	float points_y[N];
	float intercept = 0;
	float slope = 1;
	float params[] = {intercept, slope};
	make_inliers(points_x, points_y, N, params, N_PARAMS, X_MIN, X_MAX);
	float std = 0;
	int ret = add_gaussian_noise(points_y, N, std);
	ASSERT(ret == -1, 
			"with std 0, add_gaussian_noise should return -1.");
	return 0;
}

/* std <= 0 should return -1 */
int test_add_gaussian_noise_std_lt_0() {
	float points_x[N];
	float points_y[N];
	float intercept = 0;
	float slope = 1;
	float params[] = {intercept, slope};
	make_inliers(points_x, points_y, N, params, N_PARAMS, X_MIN, X_MAX);
	float std = -2;
	int ret = add_gaussian_noise(points_y, N, std);
	ASSERT(ret == -1, 
			"with std -2, add_gaussian_noise should return -1.");
	return 0;
}

/** Main runs tests in development. */
int main(){
	srand(time(NULL));
	/* ------- RUNNING TESTS FOR MAKE_INLIERS ---------- */
	printf("***** RUNNING TESTS FOR MAKE_INLIERS *****\n");
	int i = 1;
	RUN_TEST(test_make_inliers_slope_1_intercept_0, i); 
	i++;
	RUN_TEST(test_make_inliers_slope_neg1_intercept_n, i);
	i++;
	RUN_TEST(test_make_inliers_slope_float_intercept_0, i);
	i++;
	RUN_TEST(test_make_inliers_slope_1_intercept_neg_float, i);
	i++;
	RUN_TEST(test_make_inliers_slope_0_intercept_0, i);
	i++;
	RUN_TEST(test_make_inliers_qudratic_model, i);
	i++;
	RUN_TEST(test_make_inliers_edge_n_lt_2, i);
	i++;
	RUN_TEST(test_make_inliers_edge_xmin_eq_xmax, i);
	i++;
	RUN_TEST(test_make_inliers_edge_n_params_lt_2, i);
	i++;
	/* ------- RUNNING TESTS FOR ADD_GAUSSIAN_NOISE ---------- */
	printf("***** RUNNING TESTS FOR ADD_GAUSSIAN_NOISE *****\n");
	RUN_TEST(test_add_gaussian_noise_std_int, i);
	i++;
	RUN_TEST(test_add_gaussian_noise_std_float, i);
	i++;
	RUN_TEST(test_add_gaussian_noise_n_inliers_lt_2, i);
	i++;
	RUN_TEST(test_add_gaussian_noise_std_0, i);
	i++;
	RUN_TEST(test_add_gaussian_noise_std_lt_0, i);
	i++;
	return 0;
}
