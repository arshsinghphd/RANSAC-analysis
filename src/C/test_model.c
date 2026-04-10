/* =============================================================================
	Tests for model.c functions:
		eval_model
		fit_model
		find_model_inliers
		stitch_models
		points_to_line_distances
		model_error
	 
	fit_model fits a polynomial model of degree n_params - 1 to n_points data
	points using least squares via Gaussian elimination on the normal equations.
	Coefficients are stored in params at offset pos * n_params, from lowest to
	highest degree.
	 
	points_to_line_distances computes vertical distance from each point to the 
	model, storing results in distances.
============================================================================= */

#include "model.h"
#include "generator.h"
#include<stdio.h>
#include<string.h>
#include<math.h>

#include <assert.h>
#include <string.h>

#define N 10
#define N_PARAMS 2
#define MAX_PARAMS 10
#define EPSILON 1e-4
#define TRUE_SLOPE 1.0f
#define TRUE_INTERCEPT 0.0f


/* =============================================================================
	HELPERS
============================================================================= */
/* Helper function tests if two float values are almost the same. */
static void assert_almost_equal(double a, double b, const char *label) {
    if (fabs(a - b) > EPSILON) {
        printf("FAIL %s: expected %.6f got %.6f\n", label, b, a);
    } else {
        printf("PASS %s\n", label);
    }
}

static void assert_equal_delta(double a, double b, double delta, const char *label) {
    if (fabs(a - b) > delta) {
        printf("FAIL %s: expected %.6f got %.6f\n", label, b, a);
    } else {
        printf("PASS %s\n", label);
    }
}
/* Helper function tests if two int values are same. */
static void assert_equal_int(int a, int b, const char *label) {
    if (a != b) printf("FAIL %s: expected %d got %d\n", label, b, a);
    else        printf("PASS %s\n", label);
}

/* Helper function tests if all values are same in two float arrays */
static void assert_equal_float_array(float* a, float* b, int n, 
	const char* label) {
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > EPSILON) {
            printf("FAIL %s: index %d expected %f got %f\n", 
                   label, i, b[i], a[i]);
            return;
        }
    }
    printf("PASS %s\n", label);
}
/* =============================================================================
	Tests for eval_model which evaluates a polynomial model at a given x.

	Happy paths:
	    n_params = 2, line y = 1 + 2x, x = 3, result should be 7.0
	    n_params = 3, quadratic y = 1 + 2x + 0.5x^2, x = 2, result = 7.0
	    n_params = 2, x = 0, result equals intercept (a0)

	Edge cases:
	    n_params < 2, should return -1
============================================================================= */

/* n_params = 2, a0 = 1, a1 = 2. eval at x = 3 should give 7.0 */
void test_line_at_x(){
	float a0 = 1.0f;
	float a1 = 2.0f;
	float x = 3;
	float params[2] = {a0, a1};
	float result = eval_model(x, params, N_PARAMS);
	assert_almost_equal(result, 7.0f, "line_at_x");
}


/* n_params = 3, a0 = 1, a1 = 2, a2 = 0.5. eval at x = 2 should give 7.0. */
void test_quadratic_at_x() {
	float params[3] = {1.0f, 2.0f, 0.5f};
	int n_params = 3;
	float x = 2.0f;
	float result = eval_model(x, params, n_params);
	assert_almost_equal(result, 7.0f, "quadratic_at_x");
}

/* x = 0. Result should equal a0 (intercept) regardless of other params. */
void test_eval_at_zero() {
	float params[3] = {5.0, 3.0, 1.0};
	int n_params = 3;
	float x = 0.0;
	float result = eval_model(x, params, n_params);
	assert_almost_equal(result, 5.0, "eval_at_zero");
}

/* n_params = 1, should return -1. */
void test_n_params_lt_2() {
	float params[1] = {1.0};
	int n_params = 1;
	float x = 1.0f;
	float result = eval_model(x, params, n_params);
	assert_almost_equal(result, -1, "n_params_lt_2");
}


/* =============================================================================
	Tests for fit_model which fits a polynomial model using least squares via
	Gaussian elimination. Coefficients stored in params at pos * n_params. 
	params[pos * n_params + 0] = a0  (intercept)
	params[pos * n_params + 1] = a1  (slope)
	params[pos * n_params + 2] = a2  (quadratic term)

	Happy paths:
	    slope = 1, intercept = 0, n_params = 2, n_points = 10
	        exact recovery on clean data
	    slope = -1, intercept = 5, n_params = 2, n_points = 10
	        exact recovery with negative slope
	    slope = 0, intercept = 3, n_params = 2, n_points = 10
	        exact recovery for flat line
	    slope = 0.5, intercept = 0, n_params = 2, n_points = 10
	        exact recovery for fractional slope
	    slope = 1, intercept = 0, n_params = 2, n_points = 2
	        exact recovery at minimum sample size
	    quadratic a0=1, a1=2, a2=0.5, n_params = 3, n_points = 20
	        exact recovery of quadratic model

	Edge cases:
	    pos < 0,                    should return -1
	    n_points < n_params,        should return -1
	    n_params < 2,               should return -1
	    all x values equal,         should return -1 (singular matrix)
============================================================================= */

/* intercept = 0, slope = 1. */
void test_unit_slope_zero_intercept() {
    float params[2] = {0.0f, 1.0f};
    float points_x[N], points_y[N];
    float x_min = 0.0;
    float x_max = (float) N - 1;
    make_inliers(points_x, points_y, N, params, N_PARAMS, x_min, x_max);
    fit_model(points_x, points_y, N, params, N_PARAMS);
    assert_almost_equal(params[0], 0.0f, "unit_slope/intercept");
    assert_almost_equal(params[1], 1.0f, "unit_slope/slope");
}

/* intercept = 0, slope = -1. */
void test_negative_slope() {
    float params[2] = {0.0f, -1.0f};
    float points_x[N], points_y[N];
    float x_min = 0.0;
    float x_max = (float) N - 1;
    make_inliers(points_x, points_y, N, params, N_PARAMS, x_min, x_max);
    fit_model(points_x, points_y, N, params, N_PARAMS);
    assert_almost_equal(params[0], 0.0f, "negative_slope/intercept");
    assert_almost_equal(params[1], -1.0f, "negative_slope/slope");
}

/* intercept = 0, slope = 0 */
void test_zero_slope() {
    float params[2] = {0.0f, 0.0f};
    float points_x[N], points_y[N];
    float x_min = 0.0;
    float x_max = (float) N - 1;
    make_inliers(points_x, points_y, N, params, N_PARAMS, x_min, x_max);
    fit_model(points_x, points_y, N, params, N_PARAMS);
    assert_almost_equal(params[0], 0, "zero_slope/intercept");
    assert_almost_equal(params[1], 0, "zero_slope/slope");
}

/* intercept = 0, slope = 0.5 */
void test_fractional_slope() {
    float params[2] = {0.0f, 0.5f};
    float points_x[N], points_y[N];
    float x_min = 0.0;
    float x_max = (float) N - 1;
    make_inliers(points_x, points_y, N, params, N_PARAMS, x_min, x_max);
    fit_model(points_x, points_y, N, params, N_PARAMS);
    assert_almost_equal(params[0], 0, "fractional_slope/intercept");
    assert_almost_equal(params[1], 0.5, "fractional_slope/slope");
}

/* n_params = 2 */
void test_minimum_points() {
    int n = 2; 
    float params[2] = {0.0f, 1.0f};
    float points_x[n], points_y[n];
    float x_min = 0.0;
    float x_max = (float) n - 1;
    make_inliers(points_x, points_y, n, params, N_PARAMS, x_min, x_max);
    fit_model(points_x, points_y, n, params, N_PARAMS);
    assert_almost_equal(params[0], 0, "minimum_points/intercept");
    assert_almost_equal(params[1], 1, "minimum_points/slope");
}


void test_quadratic_model() {
    int n = 2; 
    float a0 = 1.0f;
    float a1 = 1.0f;
    float a2 = 1.0f;
    float params[3] = {a0, a1, a2};
    int n_params = 3;
    float points_x[n], points_y[n];
    float x_min = 0.0;
    float x_max = (float) n - 1;
    make_inliers(points_x, points_y, n, params, n_params, x_min, x_max);
    fit_model(points_x, points_y, n, params, n_params);
    assert_almost_equal(params[0], a0, "quadratic/a0");
    assert_almost_equal(params[1], a1, "quadratic/a1");
    assert_almost_equal(params[2], a2, "quadratic/a2");
}

void test_n_points_less_than_n_params() {
	int n = 10; 
    float params[2] = {0.0f, 1.0f};
    int n_params = 2;
    float points_x[n], points_y[n];
    float x_min = 0.0;
    float x_max = (float) n - 1;
    make_inliers(points_x, points_y, n, params, n_params, x_min, x_max);
    n = 1; // override to test
    int ret = fit_model(points_x, points_y, n, params, n_params);
    assert_equal_int(ret, -1, "n_points_lt_n_params");
}

void test_n_params_less_than_2() {
	int n = 10; 
    float params[2] = {0.0f, 1.0f};
    int n_params = 2;
    float points_x[n], points_y[n];
    float x_min = 0.0;
    float x_max = (float) n - 1;
    make_inliers(points_x, points_y, n, params, n_params, x_min, x_max);
    n_params = 1; // override to test
    int ret = fit_model(points_x, points_y, n, params, n_params);
    assert_equal_int(ret, -1, "n_params_lt_2");
}

void test_all_x_equal() {
    int n = 10; 
    float params[2] = {0.0f, 1.0f}; 
    int n_params = 2;
    float points_x[n], points_y[n];
    float x_min = 0.0f; 
    float x_max = 0.0f; // x_min == x_max
    make_inliers(points_x, points_y, n, params, n_params, x_min, x_max);
    int ret = fit_model(points_x, points_y, n, params, n_params);
    assert_equal_int(ret, -1, "all_x_equal");
}

/* =============================================================================
	Tests for find_model_inliers which collects inliers using vertical
	residual from the polynomial model. Points whose absolute residual is
	within threshold are appended to inliers_x and inliers_y in place.

	Happy paths:
		all points on the line, all points collected as inliers
		no points within threshold, inliers_x and inliers_y remain empty
		mixed points, only those within threshold collected
		quadratic model, inliers collected correctly

	Edge cases:
		threshold <= 0,     should return -1
		n_params < 2,       should return -1
		pos < 0,            should return -1 
============================================================================= */

/* All points on the line. No noise, all should be collected as inliers. */
void test_all_points_on_line() {
	float params[2] = {0.0f, 1.0f};
	float points_x[N], points_y[N], inliers_x[N], inliers_y[N], test_arr[N];
	float threshold = 1e-3;
	for(int i = 0; i < N; i++) {
		points_x[i] = (float) i;
		// y are 10 threshold away from linear model 
		points_y[i] = (params[0] + params[1] * points_x[i]); // no noise
		inliers_x[i] = -1.0f;
		inliers_y[i] = -1.0f;
		test_arr[i] = -1.0f;  // initial state of inliers_x, inliers_y
	}
	int n_inliers;
	int result = find_model_inliers(points_x, points_y, N, params, N_PARAMS, 
		threshold, inliers_x, inliers_y, &n_inliers);
	assert_equal_int(result, 0, "test_all_points_on_line/result");
	assert_equal_int(n_inliers, N, "test_all_points_on_line/n_inliers");
	assert_equal_float_array(inliers_x, points_x, n_inliers, 
		"test_all_points_on_line/inliers_x");
	assert_equal_float_array(inliers_y, points_y, n_inliers, 
		"test_all_points_on_line/inliers_y");
}


/* All points far from the line. inliers_x and inliers_y remain empty. */
void test_no_points_within_threshold() {
	float params[] = {0.0f, 1.0f};
	float threshold = 0.1f;
	int n = 3;
	float points_x[N], points_y[N], inliers_x[N], inliers_y[N], test_arr[N];
	for(int i = 0; i < N; i++) {
		points_x[i] = (float) i;
		// y are 10 threshold away from linear model 
		points_y[i] = (params[0] + params[1] * points_x[i]) + 10 * threshold;
		inliers_x[i] = -1.0f;
		inliers_y[i] = -1.0f;
		test_arr[i] = -1.0f;  // initial state of inliers_x, inliers_y
	}
	int n_inliers;
	int result = find_model_inliers(points_x, points_y, N, params, N_PARAMS, 
		threshold, inliers_x, inliers_y, &n_inliers);
	assert_equal_int(result, 0, "test_no_points_within_threshold/result");
	assert_equal_int(n_inliers, 0, "test_all_points_on_line/n_inliers");
	assert_equal_float_array(inliers_x, test_arr, N, 
		"test_no_points_within_threshold/result");
	assert_equal_float_array(inliers_y, test_arr, N, 
		"test_no_points_within_threshold/result");
}

/* Half points on line, half far. Only those within threshold collected. */
void test_mixed_points() {
	float params[] = {0.0f, 1.0f};
	float threshold = 0.1f;
	float points_x[N], points_y[N], inliers_x[N], inliers_y[N];
	float test_x[N], test_y[N];
	int expected_inliers = N / 2;
	for(int i = 0; i < N; i++) {
		points_x[i] = (float) i;
		if(i < expected_inliers) {
			// points are on model
			points_y[i] = params[0] + params[1] * points_x[i];
			test_x[i] = points_x[i]; // expected inliers_x
			test_y[i] = points_y[i]; // expected inliers_y
		} else {
			// y is 10 threshold away from linear model
			points_y[i] = params[0] + params[1] * points_x[i] + 10 * threshold;
			// test, like inliers, initiated at -1.
			test_x[i] = -1.0f;
			test_y[i] = -1.0f;
		}
		// all inliers initiated at unexpected value of -1.
		inliers_x[i] = -1.0f;
		inliers_y[i] = -1.0f;
	}
	int n_inliers;
	int result = find_model_inliers(points_x, points_y, N, params, N_PARAMS, 
		threshold, inliers_x, inliers_y, &n_inliers);
	assert_equal_int(result, 0, "mixed_points/result");
	assert_equal_int(n_inliers, expected_inliers, "test_all_points_on_line/n_inliers");
	assert_equal_float_array(inliers_x, test_x, N, "mixed_points/inliers_x");
	assert_equal_float_array(inliers_y, test_y, N, "mixed_points/inliers_y");
}

/* Quadratic model a0 = 1, a1 = 2, a2 = 0.5. Points on the curve are inliers. */
void test_quadratic_model_inliers() {
	float a0 = 1.0f;
	float a1 = 2.0f;
	float a2 = 0.5f;
	float params[] = {a0, a1, a2};
	int n_params = 3;
	float threshold = 0.1f;
	float points_x[N], points_y[N];
	float inliers_x[N], inliers_y[N];
	float test_x[N], test_y[N];
	for(int i = 0; i < N; i++) {
		points_x[i] = (float) i;
		// no noise
		points_y[i] = (params[0] + params[1] * points_x[i]
			+ params[2] * pow(points_x[i], 2));
		// initiate inliers
		inliers_x[i] = -1.0f;
		inliers_y[i] = -1.0f;
		// expected inliers
		test_x[i] = points_x[i];
		test_y[i] = points_y[i];
	}
	int n_inliers;
	int result = find_model_inliers(points_x, points_y, N, params, n_params,
		threshold, inliers_x, inliers_y, &n_inliers);
	assert_equal_int(result, 0, "quadratic_model_inliers/result");
	assert_equal_int(n_inliers, N, "quadratic_model_inliers/n_inliers");
	assert_equal_float_array(inliers_x, test_x, N,
		"quadratic_model_inliers/inliers_x");
	assert_equal_float_array(inliers_y, test_y, N,
		"quadratic_model_inliers/inliers_y");
}

/* threshold = 0, should return -1. */
void test_threshold_zero() {
	float params[2] = {0.0, 1.0};
    float points_x[N], points_y[N], inliers_x[N], inliers_y[N];
    float threshold = 0;
    int n_inliers;
    int res = find_model_inliers(points_x, points_y, N, params, N_PARAMS,
    	threshold, inliers_x, inliers_y, &n_inliers);
    assert_equal_int(res, -1, "threshold_zero/result");
}

/* threshold = -1, should return -1. */
void test_threshold_negative() {
	float params[2] = {0.0, 1.0};
    float points_x[N], points_y[N], inliers_x[N], inliers_y[N];
    float threshold = -1;
    int n_inliers;
    int res = find_model_inliers(points_x, points_y, N, params, N_PARAMS,
    	threshold, inliers_x, inliers_y, &n_inliers);
    assert_equal_int(res, -1, "threshold_negative/result");
}

/* n_params = 1, should return -1. */
void test_n_params_less_than_2_inliers() {
	float params[2] = {0.0, 1.0};
    float points_x[N], points_y[N], inliers_x[N], inliers_y[N];
    float threshold = -1;
    int n_params = 1;
    int n_inliers;
    int res = find_model_inliers(points_x, points_y, N, params, n_params,
    	threshold, inliers_x, inliers_y, &n_inliers);
    assert_equal_int(res, -1, "n_params_less_than_2_inliers/result");
}


/*==============================================================================
	Tests for stitch_models which combines inliers from two overlapping graphs
    into a single refined model using params1 and params2.

    Happy paths:
        two clean overlapping graphs with same true model
            combined fit should recover true slope and intercept exactly
        two noisy overlapping graphs with same true model
            combined fit should recover true model within tolerance

    Edge cases:
        n1 < n_params,      should return -1
        n2 < n_params,      should return -1
        threshold <= 0,     should return -1
        no inliers graph 1, should return -1
        no inliers graph 2, should return -1
==============================================================================*/

/* ---------------------------------------------------------------
 * Helper: fills points_x, points_y with n inliers on the true
 * model over [x_min, x_max], optionally adding gaussian noise.
 * noise_std = 0 means no noise.
 * --------------------------------------------------------------- */
static void _make_graph(float* points_x, float* points_y, int n,
                        float x_min, float x_max, float noise_std) {
    float params[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    make_inliers(points_x, points_y, n, params, N_PARAMS, x_min, x_max);
    if (noise_std > 0)
        add_gaussian_noise(points_y, n, noise_std);
}


/* ---------------------------------------------------------------
 * Helper: runs stitch_models and returns result.
 * --------------------------------------------------------------- */
static int _stitch(float* px1, float* py1, int n1, float* params1,
                   float* px2, float* py2, int n2, float* params2,
                   float* params, float threshold) {
    return stitch_models(px1, py1, n1, params1,
                         px2, py2, n2, params2,
                         params, N_PARAMS, threshold);
}

/* Two clean overlapping graphs with same true model.
 * Combined fit should recover true model exactly. */
void test_clean_overlapping_graphs() {
    int n = 100, n1 = 55, n2 = 55;
    float points_x1[n1], points_y1[n1];
    float points_x2[n2], points_y2[n2];
    float params1[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    float params2[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    // graph 1: x in [0, n1 - 1], no noise
    _make_graph(points_x1, points_y1, n1, 0, n1 - 1, 0);
    // graph 2: x in [n - n2, n - 1], overlap of 10 points, no noise
    _make_graph(points_x2, points_y2, n2, n - n2, n - 1, 0);
    float params[N_PARAMS];
    int result = _stitch(points_x1, points_y1, n1, params1,
                         points_x2, points_y2, n2, params2,
                         params, 0.1f);
    assert_equal_int(result, 0, "clean_overlapping_graphs/result");
    assert_almost_equal(params[0], TRUE_INTERCEPT,
        "clean_overlapping_graphs/intercept");
    assert_almost_equal(params[1], TRUE_SLOPE,
        "clean_overlapping_graphs/slope");
}


/* Two noisy overlapping graphs with same true model.
 * Combined fit should recover true model within threshold. */
void test_noisy_overlapping_graphs() {
    int n = 100, n1 = 55, n2 = 55;
    float std = 1.0f, threshold = 3.0f;
    float points_x1[n1], points_y1[n1];
    float points_x2[n2], points_y2[n2];
    float params1[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    float params2[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    // graph 1: x in [0, n1 - 1], gaussian noise std = 1
    _make_graph(points_x1, points_y1, n1, 0, n1 - 1, std);
    // graph 2: x in [n - n2, n - 1], overlap of 10 points, gaussian noise std = 1
    _make_graph(points_x2, points_y2, n2, n - n2, n - 1, std);
    float params[N_PARAMS];
    int result = _stitch(points_x1, points_y1, n1, params1,
                         points_x2, points_y2, n2, params2,
                         params, threshold);
    assert_equal_int(result, 0, "noisy_overlapping_graphs/result");
    assert_equal_delta(params[0], TRUE_INTERCEPT, threshold,
        "noisy_overlapping_graphs/intercept");
    assert_equal_delta(params[1], TRUE_SLOPE, threshold,
        "noisy_overlapping_graphs/slope");
}


/* n1 = 1, should return -1. */
void test_n1_lt_n_params() {
    int n = 100, n1 = 55, n2 = 55;
    float points_x1[n1], points_y1[n1];
    float points_x2[n2], points_y2[n2];
    float params1[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    float params2[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    _make_graph(points_x1, points_y1, n1, 0, n1 - 1, 1.0f);
    _make_graph(points_x2, points_y2, n2, n - n2, n - 1, 1.0f);
    float params[N_PARAMS];
    // overwrite n1 to trigger error
    int result = _stitch(points_x1, points_y1, 1, params1,
                         points_x2, points_y2, n2, params2,
                         params, 0.1f);
    assert_equal_int(result, -1, "n1_lt_n_params/result");
}


/* n2 = 1, should return -1. */
void test_n2_lt_n_params() {
    int n = 100, n1 = 55, n2 = 55;
    float points_x1[n1], points_y1[n1];
    float points_x2[n2], points_y2[n2];
    float params1[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    float params2[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    _make_graph(points_x1, points_y1, n1, 0, n1 - 1, 1.0f);
    _make_graph(points_x2, points_y2, n2, n - n2, n - 1, 1.0f);
    float params[N_PARAMS];
    // overwrite n2 to trigger error
    int result = _stitch(points_x1, points_y1, n1, params1,
                         points_x2, points_y2, 1, params2,
                         params, 0.1f);
    assert_equal_int(result, -1, "n2_lt_n_params/result");
}


/* threshold = 0, should return -1. */
void test_threshold_zero_stitch_models() {
    int n = 100, n1 = 55, n2 = 55;
    float points_x1[n1], points_y1[n1];
    float points_x2[n2], points_y2[n2];
    float params1[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    float params2[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    _make_graph(points_x1, points_y1, n1, 0, n1 - 1, 1.0f);
    _make_graph(points_x2, points_y2, n2, n - n2, n - 1, 1.0f);
    float params[N_PARAMS];
    int result = _stitch(points_x1, points_y1, n1, params1,
                         points_x2, points_y2, n2, params2,
                         params, 0.0f);
    assert_equal_int(result, -1, "threshold_zero/result");
}


/* threshold = -1, should return -1. */
void test_threshold_negative_stitch_models() {
    int n = 100, n1 = 55, n2 = 55;
    float points_x1[n1], points_y1[n1];
    float points_x2[n2], points_y2[n2];
    float params1[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    float params2[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    _make_graph(points_x1, points_y1, n1, 0, n1 - 1, 1.0f);
    _make_graph(points_x2, points_y2, n2, n - n2, n - 1, 1.0f);
    float params[N_PARAMS];
    int result = _stitch(points_x1, points_y1, n1, params1,
                         points_x2, points_y2, n2, params2,
                         params, -1.0f);
    assert_equal_int(result, -1, "threshold_negative/result");
}


/* Threshold too small for noisy graph 1. Should return -1. */
void test_no_inliers_graph1() {
    int n = 100, n1 = 55, n2 = 55;
    float points_x1[n1], points_y1[n1];
    float points_x2[n2], points_y2[n2];
    float params1[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    float params2[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    // graph 1 has noise, graph 2 is clean
    _make_graph(points_x1, points_y1, n1, 0, n1 - 1, 1.0f);
    _make_graph(points_x2, points_y2, n2, n - n2, n - 1, 0.0f);
    float params[N_PARAMS];
    // threshold too small for noisy graph 1
    int result = _stitch(points_x1, points_y1, n1, params1,
                         points_x2, points_y2, n2, params2,
                         params, 1e-9f);
    assert_equal_int(result, -1, "no_inliers_graph1/result");
}


/* Threshold too small for noisy graph 2. Should return -1. */
void test_no_inliers_graph2() {
    int n = 100, n1 = 55, n2 = 55;
    float points_x1[n1], points_y1[n1];
    float points_x2[n2], points_y2[n2];
    float params1[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    float params2[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    // graph 1 is clean, graph 2 has noise
    _make_graph(points_x1, points_y1, n1, 0, n1 - 1, 0.0f);
    _make_graph(points_x2, points_y2, n2, n - n2, n - 1, 1.0f);
    float params[N_PARAMS];
    // threshold too small for noisy graph 2
    int result = _stitch(points_x1, points_y1, n1, params1,
                         points_x2, points_y2, n2, params2,
                         params, 1e-9f);
    assert_equal_int(result, -1, "no_inliers_graph2/result");
}


/*==============================================================================
Tests for points_to_line_distances which computes perpendicular distance
    from each point to a line defined by slope and intercept.

        distances[i] = |slope * points_x[i] - points_y[i] + intercept|
                       / sqrt(1 + slope squared)

    Happy paths:
        all points on the line, all distances equal 0.0
        points 1 unit above the line, distance = 1 / sqrt(1 + slope squared)
        slope = 0, points 2 units above, distance = 2.0
        slope = -1, points 1 unit above, distances positive

    Edge cases:
        n_points = 0, should return -1

==============================================================================*/




/*==============================================================================
  MAIN 
==============================================================================*/
int main() {
	printf("***** RUNNING TESTS FOR FIT_MODEL *****\n");
    test_unit_slope_zero_intercept();
    test_negative_slope();
    test_zero_slope();
    test_fractional_slope();
    test_minimum_points();
    test_quadratic_model();
    test_n_points_less_than_n_params();
    test_n_params_less_than_2();
    test_all_x_equal();
	printf("***** RUNNING TESTS FOR EVAL_MODEL *****\n");
    test_line_at_x();
    test_quadratic_at_x();
    test_eval_at_zero();
    test_n_params_lt_2();
    printf("***** RUNNING TESTS FOR FIND_MODEL_INLIERS *****\n");
	test_all_points_on_line();
	test_no_points_within_threshold();
	test_mixed_points();
	test_quadratic_model_inliers();
	test_threshold_zero();
	test_threshold_negative();
	test_n_params_less_than_2_inliers();
	printf("***** RUNNING TESTS FOR STITCH_MODELS *****\n");
	test_clean_overlapping_graphs();
	test_noisy_overlapping_graphs();
	test_n1_lt_n_params();
	test_n2_lt_n_params();
	test_threshold_zero_stitch_models();
	test_threshold_negative_stitch_models();
	test_no_inliers_graph1();
    return 0;
}
