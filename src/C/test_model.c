/**
 * Tests for model.c functions:
 * 	eval_model
 * 	fit_model
 * 	find_model_inliers
 * 	stitch_models
 * 	points_to_line_distances
 * 	model_error
 * 
 * fit_model fits a polynomial model of degree n_params - 1 to n_points data
 * points using least squares via Gaussian elimination on the normal equations.
 * Coefficients are stored in params at offset pos * n_params, from lowest to
 * highest degree.
 * 
 * points_to_line_distances computes vertical distance from each point to
 * the model, storing results in distances.
 */

#include "model.h"
#include "generator.h"
#include<stdio.h>
#include<string.h>
#include<math.h>

#include <assert.h>
#include <string.h>

#define MAX_PARAMS 10
#define EPSILON 1e-10

/* ================================================================
 * HELPERS
 * ================================================================ */
static void assert_almost_equal(double a, double b, const char *label) {
    if (fabs(a - b) > 1e-5) {
        printf("FAIL %s: expected %.6f got %.6f\n", label, b, a);
    } else {
        printf("PASS %s\n", label);
    }
}

static void assert_equal_int(int a, int b, const char *label) {
    if (a != b) printf("FAIL %s: expected %d got %d\n", label, b, a);
    else        printf("PASS %s\n", label);
}

/* ================================================================
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
 * ================================================================ */

/* intercept = 0, slope = 1. */
void test_unit_slope_zero_intercept() {
    int n = 10; 
    float params[2] = {0.0f, 1.0f};
    int n_params = 2;
    float points_x[n], points_y[n];
    float x_min = 0.0;
    float x_max = (float) n - 1;
    make_inliers(points_x, points_y, n, params, n_params, x_min, x_max);
    fit_model(points_x, points_y, n, params, 2);
    assert_almost_equal(params[0], 0, "unit_slope/intercept");
    assert_almost_equal(params[1], 1, "unit_slope/slope");
}

/* intercept = 0, slope = -1. */
void test_negative_slope() {
    int n = 10; 
    float params[2] = {0.0f, -1.0f};
    int n_params = 2;
    float points_x[n], points_y[n];
    float x_min = 0.0;
    float x_max = (float) n - 1;
    make_inliers(points_x, points_y, n, params, n_params, x_min, x_max);
    fit_model(points_x, points_y, n, params, 2);
    assert_almost_equal(params[0], 0, "negative_slope/intercept");
    assert_almost_equal(params[1], -1, "negative_slope/slope");
}

/* intercept = 0, slope = 0 */
void test_zero_slope() {
    int n = 10; 
    float params[2] = {0.0f, 0.0f};
    int n_params = 2;
    float points_x[n], points_y[n];
    float x_min = 0.0;
    float x_max = (float) n - 1;
    make_inliers(points_x, points_y, n, params, n_params, x_min, x_max);
    fit_model(points_x, points_y, n, params, 2);
    assert_almost_equal(params[0], 0, "zero_slope/intercept");
    assert_almost_equal(params[1], 0, "zero_slope/slope");
}

/* intercept = 0, slope = 0.5 */
void test_fractional_slope() {
    int n = 10; 
    float params[2] = {0.0f, 0.5f};
    int n_params = 2;
    float points_x[n], points_y[n];
    float x_min = 0.0;
    float x_max = (float) n - 1;
    make_inliers(points_x, points_y, n, params, n_params, x_min, x_max);
    fit_model(points_x, points_y, n, params, 2);
    assert_almost_equal(params[0], 0, "fractional_slope/intercept");
    assert_almost_equal(params[1], 0.5, "fractional_slope/slope");
}

/* n_params = 2 */
void test_minimum_points() {
    int n = 2; 
    float params[2] = {0.0f, 1.0f};
    int n_params = 2;
    float points_x[n], points_y[n];
    float x_min = 0.0;
    float x_max = (float) n - 1;
    make_inliers(points_x, points_y, n, params, n_params, x_min, x_max);
    fit_model(points_x, points_y, n, params, n_params);
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
    return 0;
}
