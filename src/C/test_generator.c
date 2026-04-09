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
#include<string.h>
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

// for make inliers, noise
#define N 10
#define X_MIN 0.0
#define X_MAX 9.0
#define N_PARAMS 2
#define STD 2.0

// additional for outlier testing
#define N_INLIERS 90
#define N_OUTLIERS 10
#define N_POINTS 100

// additional for structural bias testing 
#define PR 1

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
	add_gaussian_noise(points_y, N, STD);
	int changed;
	for(int i = 0; i < N; i++){
		changed = 0;
		if(fabs(points_y[i] - (intercept + i * slope)) > 1e-6)
        	changed = 1;
        ASSERT(changed, "std 2: no noise was added");
		ASSERT(fabs(points_y[i] - (intercept + i * slope)) < 3 * STD, 
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
	int ret = add_gaussian_noise(points_y, n, STD);
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


/* ------------ TEST ADD_OUTLIERS ------------ */
/**
 * Tests happy paths and edge cases for add_outliers.
 * add_outliers appends points guaranteed to lie outside the inlier band 
 * defined by the true polynomial model and noise_std, using vertical
 * residual consistent with find_model_inliers. The inlier band is:
 * 		|y - model(x)| < 2 * noise_std
 * 
 * Happy paths:
 * inlier points unchanged after call
 * n_outliers points appended at the end, 
 * all outside inlier band
 * 
 * Special cases:
 * n_outliers == 0; data unchanged, return 0
 * y_min == y_max; should return 0; inliers unchanged; add outliers outside band
 * 
 * Edge cases:
 * n_inliers < 0               	should return -1
 * n_outliers < 0              	should return -1
 * n_inliers + n_outliers < 2  	should return -1
 * noise_std <= 0              	should return -1
 * n_params < 2                	should return -1
 * x_min == x_max				should return -1
 */

/** Test happy path. */
int test_add_outliers_happy_path(){
	float points_x[N_POINTS];
	float points_y[N_POINTS];
	float a = 0;
	float b = 1;
	float params[] = {a, b};
	// fill N_INLIERS
	make_inliers(points_x, points_y, N_INLIERS, params, N_PARAMS, X_MIN, X_MAX);
	// add noise to N_INLIERS
	add_gaussian_noise(points_y, N_INLIERS, STD);
	// make a copy of points_y for testing
	float copy_y[N_INLIERS];
	memcpy(copy_y, points_y, N_INLIERS * sizeof(float));
	// add N_OUTLIERS
	add_outliers(points_x, points_y, N_INLIERS, N_OUTLIERS, params, N_PARAMS, STD);
	for(int i = 0; i < N_POINTS; i++){
		if(i < N_INLIERS) {
			ASSERT(fabs(points_y[i] - copy_y[i]) < 1e-6, 
				"happy path: inliers changed.");
		} else {
			ASSERT(fabs(points_y[i] - (a + b * points_x[i])) > 2 * STD, 
				"happy path: outlier is not outside the band.");
		}
	}
	return 0;
}

/** Test special case: n_outliers = 0. */
int test_add_outliers_n_outliers_0(){
	float points_x[N_POINTS];
	float points_y[N_POINTS];
	float a = 0;
	float b = 1;
	float params[] = {a, b};
	// fill N_INLIERS
	make_inliers(points_x, points_y, N_POINTS, params, N_PARAMS, X_MIN, X_MAX);
	// add 0 N_OUTLIERS
	int ret = add_outliers(points_x, points_y, N_POINTS, 0, params, N_PARAMS, STD);
	ASSERT(ret == 0, "n_outliers 0, return should be 0.");
	return 0;
}

/** Test edge cases: n_inliers < 0, should return -1. */
int test_add_outliers_n_inliers_lt_0(){
	float points_x[N_INLIERS];
	float points_y[N_INLIERS];
	float a = 0;
	float b = 1;
	float params[] = {a, b};
	// fill N_INLIERS
	make_inliers(points_x, points_y, N_INLIERS, params, N_PARAMS, X_MIN, X_MAX);
	// pass -1 for n_inliers
	int ret = add_outliers(points_x, points_y, -1, N_OUTLIERS, params, N_PARAMS, STD);
	ASSERT(ret == -1, "n_inliers -1, return should be -1.");
	return 0;
}

/** Test edge cases: n_outliers < 0, should return -1. */
int test_add_outliers_n_outliers_lt_0(){
	float points_x[N_INLIERS];
	float points_y[N_INLIERS];
	float a = 0;
	float b = 1;
	float params[] = {a, b};
	// fill N_INLIERS
	make_inliers(points_x, points_y, N_INLIERS, params, N_PARAMS, X_MIN, X_MAX);
	// pass -1 for n_outliers
	int ret = add_outliers(points_x, points_y, N_INLIERS, -1, params, N_PARAMS, STD);
	ASSERT(ret == -1, "n_outliers -1, return should be -1.");
	return 0;
}

/** Test edge cases: n_inliers + n_inliers < 2, should return -1. */
int test_add_outliers_n_points_lt_2(){
	float points_x[N_INLIERS];
	float points_y[N_INLIERS];
	float a = 0;
	float b = 1;
	float params[] = {a, b};
	// fill N_INLIERS
	make_inliers(points_x, points_y, N_INLIERS, params, N_PARAMS, X_MIN, X_MAX);
	// pass 0 for inliers and 1 for n_outliers n_points < 2
	int ret = add_outliers(points_x, points_y, 0, 1, params, N_PARAMS, STD);
	ASSERT(ret == -1, "n_points 1, return should be -1.");
	return 0;
}

/** Test edge cases: noise_std == 0, should return -1. */
int test_add_outliers_std_0() {
	float points_x[N_INLIERS];
	float points_y[N_INLIERS];
	float a = 0;
	float b = 1;
	float params[] = {a, b};
	// fill N_INLIERS
	make_inliers(points_x, points_y, N_INLIERS, params, N_PARAMS, X_MIN, X_MAX);
	// pass 0 for noise_std
	int ret = add_outliers(points_x, points_y, N_INLIERS, N_OUTLIERS, 
		params, N_PARAMS, 0);
	ASSERT(ret == -1, "noise_std 0, return should be -1.");
	return 0;
}

/** Test edge cases: noise_std < 0, should return -1. */
int test_add_outliers_std_lt_0() {
	float points_x[N_INLIERS];
	float points_y[N_INLIERS];
	float a = 0;
	float b = 1;
	float params[] = {a, b};
	// fill N_INLIERS
	make_inliers(points_x, points_y, N_INLIERS, params, N_PARAMS, X_MIN, X_MAX);
	// pass 0 for noise_std
	int ret = add_outliers(points_x, points_y, N_INLIERS, N_OUTLIERS, 
		params, N_PARAMS, -1);
	ASSERT(ret == -1, "noise_std -1, return should be -1.");
	return 0;
}

/** Test edge cases: n_params < 2, should return -1. */
int test_add_outliers_n_params_lt_2() {
	float points_x[N_INLIERS];
	float points_y[N_INLIERS];
	float a = 0;
	float b = 1;
	float params[] = {a, b};
	// fill N_INLIERS
	make_inliers(points_x, points_y, N_INLIERS, params, N_PARAMS, X_MIN, X_MAX);
	// pass 1 for n_params
	int ret = add_outliers(points_x, points_y, N_INLIERS, N_OUTLIERS, 
		params, 1, STD);
	ASSERT(ret == -1, "n_params 1, return should be -1.");
	return 0;
}

/** Test edge cases: x_min == x_max, should return -1. */
int test_add_outliers_x_min_eq_x_max() {
	float points_x[N_INLIERS];
	float points_y[N_INLIERS];
	float a = 0;
	float b = 1;
	float params[] = {a, b};
	// fill N_INLIERS
	make_inliers(points_x, points_y, N_INLIERS, params, N_PARAMS, X_MIN, X_MAX);
	// overwrite points_x such that x_min == x_max
	for (int i = 0; i < N_INLIERS; i++) {
		points_x[i] = X_MAX;
	}
	int ret = add_outliers(points_x, points_y, N_INLIERS, N_OUTLIERS, 
		params, N_PARAMS, STD);
	ASSERT(ret == -1, "x_min = x_max, return should be -1.");
	return 0;
}

/** Test edge cases: y_min == y_max, should return 0. */
int test_add_outliers_y_min_eq_y_max() {
	float points_x[N_POINTS];
	float points_y[N_POINTS];
	float a = 0;
	float b = 0; // slope is 0, y_min = y_max
	float params[] = {a, b};
	// fill N_INLIERS
	make_inliers(points_x, points_y, N_INLIERS, params, N_PARAMS, X_MIN, X_MAX);
	// make a copy of points_y for testing
	float copy_y[N_INLIERS];
	memcpy(copy_y, points_y, N_INLIERS * sizeof(float));
	// add outliers
	int ret = add_outliers(points_x, points_y, N_INLIERS, N_OUTLIERS, 
		params, N_PARAMS, STD);
	ASSERT(ret == 0, "y_min = y_max, return should be 0.");
	for(int i = 0; i < N_POINTS; i++){
		if(i < N_INLIERS) {
			ASSERT(fabs(points_y[i] - copy_y[i]) < 1e-6, 
				"y_min == y_max: inliers changed.");
		} else {
			ASSERT(fabs(points_y[i] - (a + b * points_x[i])) > 2 * STD, 
				"y_min == y_max: outlier is not outside the band.");
		}
	}
	return 0;
}

/* ------------ TEST ADD_STRUCTURAL_BIAS ------------ */
/**
 * Tests happy paths and edge cases for add_structural_bias.
 * 
 * Happy paths:
 * bias_fn = lambda x: 0           points_y unchanged
 * bias_fn = lambda x: 1.0         points_y[i] increased by 1.0
 * bias_fn = lambda x: 0.5 * x     points_y[i] increased by 0.5 * x
 * bias_fn = lambda x: sin(x)      points_y[i] increased by sin(x)
 * 
 * Edge cases:
 * bias_fn is NULL 	should return -1
 * pr < 0 			should return -1
 * pr > 1 			should return -1
 * n_inliers < 2   	should return -1
 */

static float _bias_const_0(float x) {
	return 0.0f;
}

static float _bias_const_1(float x) {
	return 1.0f;
}

static float _bias_linear(float x) {
	return 0.5 * x;
}

static float _bias_periodic(float x) {
	return sin(x);
}

/** Test happy path: bias_fn = _bias_const_0; points_y unchanged. */
int test_add_structural_bias_const_0(){
	float points_x[N_POINTS];
	float points_y[N_POINTS];
	float a = 0;
	float b = 1;
	float params[] = {a, b};
	// fill N_INLIERS
	make_inliers(points_x, points_y, N_POINTS, params, N_PARAMS, X_MIN, X_MAX);
	// make a copy of points_y for testing
	float copy_y[N_POINTS];
	memcpy(copy_y, points_y, N_POINTS * sizeof(float));
	// add structural bias
	int ret = add_structural_bias(points_y, points_x, N_POINTS, PR, _bias_const_0);
	ASSERT(ret == 0, "_bias_const_0, return should be 0.");
	for(int i = 0; i < N_POINTS; i++){
		ASSERT(fabs(points_y[i] - copy_y[i]) < 1e-6, 
			"_bias_const_0: points changed.");
	}
	return 0;
}

/** Test happy path: bias_fn = _bias_const_1; all points_y change by 1. */
int test_add_structural_bias_const_1(){
	float points_x[N_POINTS];
	float points_y[N_POINTS];
	float a = 0;
	float b = 1;
	float params[] = {a, b};
	// fill N_INLIERS
	make_inliers(points_x, points_y, N_POINTS, params, N_PARAMS, X_MIN, X_MAX);
	// make a copy of points_y for testing
	float copy_y[N_POINTS];
	memcpy(copy_y, points_y, N_POINTS * sizeof(float));
	// add structural bias
	int ret = add_structural_bias(points_y, points_x, N_POINTS, PR, _bias_const_1);
	ASSERT(ret == 0, "_bias_const_1, return should be 0.");
	for(int i = 0; i < N_POINTS; i++){
		ASSERT(fabs(points_y[i] - copy_y[i] - 1.0) < 1e-4, 
			"_bias_const_1: points_y error");
	}
	return 0;
}

/** Test happy path: bias_fn = _bias_linear; all points_y change by 0.5 * x. */
int test_add_structural_bias_linear(){
	float points_x[N_POINTS];
	float points_y[N_POINTS];
	float a = 0;
	float b = 1;
	float params[] = {a, b};
	// fill N_INLIERS
	make_inliers(points_x, points_y, N_POINTS, params, N_PARAMS, X_MIN, X_MAX);
	// make a copy of points_y for testing
	float copy_y[N_POINTS];
	memcpy(copy_y, points_y, N_POINTS * sizeof(float));
	// add structural bias
	int ret = add_structural_bias(points_y, points_x, N_POINTS, PR, _bias_linear);
	ASSERT(ret == 0, "_bias_linear, return should be 0.");
	for(int i = 0; i < N_POINTS; i++){
		ASSERT(fabs(points_y[i] - copy_y[i] - 0.5 * points_x[i]) < 1e-6, 
			"_bias_linear: points_y error");
	}
	return 0;
}

/** Test happy path: bias_fn = _bias_linear; all points_y change by 0.5 * x. */
int test_add_structural_bias_periodic(){
	float points_x[N_POINTS];
	float points_y[N_POINTS];
	float a = 0;
	float b = 1;
	float params[] = {a, b};
	// fill N_INLIERS
	make_inliers(points_x, points_y, N_POINTS, params, N_PARAMS, X_MIN, X_MAX);
	// make a copy of points_y for testing
	float copy_y[N_POINTS];
	memcpy(copy_y, points_y, N_POINTS * sizeof(float));
	// add structural bias
	int ret = add_structural_bias(points_y, points_x, N_POINTS, PR, _bias_periodic);
	ASSERT(ret == 0, "_bias_linear, return should be 0.");
	for(int i = 0; i < N_POINTS; i++) {
		ASSERT(fabs(points_y[i] - copy_y[i] - sin(points_x[i])) < 1e-6, 
			"_bias_periodic: points_y error");
	}
	return 0;
}

/* Test edge case: bias_fn is NULL 	should return -1 */
int test_add_structural_bias_null(){
	float points_x[N_POINTS];
	float points_y[N_POINTS];
	float a = 0;
	float b = 1;
	float params[] = {a, b};
	// fill N_INLIERS
	make_inliers(points_x, points_y, N_POINTS, params, N_PARAMS, X_MIN, X_MAX);
	// make a copy of points_y for testing
	float copy_y[N_POINTS];
	memcpy(copy_y, points_y, N_POINTS * sizeof(float));
	// add structural bias
	int ret = add_structural_bias(points_y, points_x, N_POINTS, PR, NULL);
	ASSERT(ret == -1, "bias = NULL, return should be -1.");
	return 0;
}

/* Test edge case: pr < 0 			should return -1 */
int test_add_structural_bias_pr_lt_0(){
	float points_x[N_POINTS];
	float points_y[N_POINTS];
	float a = 0;
	float b = 1;
	float params[] = {a, b};
	// fill N_INLIERS
	make_inliers(points_x, points_y, N_POINTS, params, N_PARAMS, X_MIN, X_MAX);
	// make a copy of points_y for testing
	float copy_y[N_POINTS];
	memcpy(copy_y, points_y, N_POINTS * sizeof(float));
	// add structural bias, PR = -1
	int ret = add_structural_bias(points_y, points_x, N_POINTS, -1, NULL);
	ASSERT(ret == -1, "pr = -1, return should be -1.");
	return 0;
}

/* Test edge case: pr > 1 			should return -1 */
int test_add_structural_bias_pr_gt_1(){
	float points_x[N_POINTS];
	float points_y[N_POINTS];
	float a = 0;
	float b = 1;
	float params[] = {a, b};
	// fill N_INLIERS
	make_inliers(points_x, points_y, N_POINTS, params, N_PARAMS, X_MIN, X_MAX);
	// make a copy of points_y for testing
	float copy_y[N_POINTS];
	memcpy(copy_y, points_y, N_POINTS * sizeof(float));
	// add structural bias, PR = 1.1
	int ret = add_structural_bias(points_y, points_x, N_POINTS, 1.1, NULL);
	ASSERT(ret == -1, "pr = 1.1, return should be -1.");
	return 0;
}

/* Test edge case: n_inliers < 2   	should return -1 */
int test_add_structural_bias_n_inliers_1(){
	float points_x[N_POINTS];
	float points_y[N_POINTS];
	float a = 0;
	float b = 1;
	float params[] = {a, b};
	// fill N_INLIERS
	make_inliers(points_x, points_y, N_POINTS, params, N_PARAMS, X_MIN, X_MAX);
	// make a copy of points_y for testing
	float copy_y[N_POINTS];
	memcpy(copy_y, points_y, N_POINTS * sizeof(float));
	// add structural bias, PR = 1.1
	int ret = add_structural_bias(points_y, points_x, 1, PR, NULL);
	ASSERT(ret == -1, "n_inliers = 1, return should be -1.");
	return 0;
}


/* ---------- MAIN ---------- */
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
	/* ------- RUNNING TESTS FOR ADD_OUTLIERS ---------- */
	printf("***** RUNNING TESTS FOR ADD_OUTLIERS *****\n");
	RUN_TEST(test_add_outliers_happy_path, i);
	i++;
	RUN_TEST(test_add_outliers_n_outliers_0, i);
	i++;
	RUN_TEST(test_add_outliers_n_inliers_lt_0, i);
	i++;
	RUN_TEST(test_add_outliers_n_outliers_lt_0, i);
	i++;
	RUN_TEST(test_add_outliers_n_points_lt_2, i);
	i++;
	RUN_TEST(test_add_outliers_std_0, i);
	i++;
	RUN_TEST(test_add_outliers_std_lt_0, i);
	i++;
	RUN_TEST(test_add_outliers_n_params_lt_2, i);
	i++;
	RUN_TEST(test_add_outliers_x_min_eq_x_max, i);
	i++;
	RUN_TEST(test_add_outliers_y_min_eq_y_max, i);
	i++;
	/* ------- RUNNING TESTS FOR ADD_OUTLIERS ---------- */
	printf("***** RUNNING TESTS FOR ADD_STRUCTURAL_BIAS *****\n");
	RUN_TEST(test_add_structural_bias_const_0, i);
	i++;
	RUN_TEST(test_add_structural_bias_const_1, i);
	i++;
	RUN_TEST(test_add_structural_bias_linear, i);
	i++;
	RUN_TEST(test_add_structural_bias_periodic, i);
	i++;
	RUN_TEST(test_add_structural_bias_null, i);
	i++;
	RUN_TEST(test_add_structural_bias_null, i);
	i++;
	RUN_TEST(test_add_structural_bias_pr_lt_0, i);
	i++;
	RUN_TEST(test_add_structural_bias_pr_gt_1, i);
	i++;
	RUN_TEST(test_add_structural_bias_n_inliers_1, i);
	i++;
	//RUN_TEST(, i);
	//i++;
	return 0;
}


