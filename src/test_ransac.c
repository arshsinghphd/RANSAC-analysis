/**
 * Tests for ransac.c functions:
 *     fisher_yates
 *     estimate_epsilon
 *     compute_t
 *     compute_k
 *     compute_d
 *     ransac
 *
 * estimate_epsilon estimates the outlier fraction from the residual
 * distribution of a preliminary least squares fit using vertical
 * residuals from eval_model.
 *
 * compute_t estimates the inlier threshold as mean + 2 * std of
 * vertical residuals from a preliminary least squares fit, consistent
 * with the recommendation of Fischler and Bolles (1981).
 *
 * compute_k computes the required number of RANSAC iterations from
 * epsilon and n_params using the analytical formula at failure
 * probability p = 0.01.
 *
 * compute_d computes the expected inlier count as
 * floor((1 - epsilon) * n_points), consistent with the same epsilon
 * used to compute k.
 *
 * ransac finds the best fitting polynomial model of degree n_params - 1
 * from noisy data containing outliers using Random Sample Consensus.
 * It randomly samples n_params points, fits a model via fit_model,
 * counts inliers within threshold vertical residual using
 * find_model_inliers, and repeats k_resample times or until
 * expected_inliers are found. Results are stored in return_array.
 *
 * return_array layout:
 *     return_array[0]               number of inliers in best model
 *     return_array[1]               number of iterations actually run
 *     return_array[2..2+n_params-1] best model params (a0, a1, ...)
 *                                   from lowest to highest degree
 */

#include "ransac.h"
#include "generator.h"
#include "model.h"

#include<math.h>
#include<time.h>
#include<stdio.h>
#include<stdlib.h>

/* Constants */
#define N 100
#define X_MIN 0.0f
#define X_MAX 99.0f
#define TRUE_SLOPE 2.0f
#define TRUE_INTERCEPT 5.0f
#define NOISE_STD 0.5f
#define N_PARAMS 2
#define EPSILON 1e-5
#define FAIL_PR 0.01

/* =============================================================================
 * HELPERS
============================================================================= */


/**
 * Asserts |a - b| < EPSILON. Prints PASS or FAIL with label.
 */
static void assert_almost_equal(float a, float b, const char* label) {
    if (fabsf(a - b) > EPSILON)
        printf("FAIL %s: expected %.6f got %.6f\n", label, b, a);
    else
        printf("PASS %s\n", label);
}


/**
 * Asserts |a - b| <= delta. Prints PASS or FAIL with label and delta.
 * Use for noisy data tests where tolerance is not a fixed EPSILON.
 */
static void assert_equal_delta(float a, float b, float delta,
                                const char* label) {
    if (fabsf(a - b) > delta)
        printf("FAIL %s: expected %.6f got %.6f delta %.6f\n",
               label, b, a, delta);
    else
        printf("PASS %s\n", label);
}


/**
 * Asserts a == b (integer equality). Prints PASS or FAIL with label.
 */
static void assert_equal_int(int a, int b, const char* label) {
    if (a != b) printf("FAIL %s: expected %d got %d\n", label, b, a);
    else        printf("PASS %s\n", label);
}


/** ----------------------------------------------------------------------------
 * Asserts all elements of a equal corresponding elements of b within
 * EPSILON. Reports the first failing index and returns immediately.
 *
 * Params:
 *     a       array of n floats, actual values
 *     b       array of n floats, expected values
 *     n       int, number of elements to compare
 *     label   string, printed on pass or fail
 ----------------------------------------------------------------------------*/
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


/** ----------------------------------------------------------------------------
 * Fills points_x and points_y with n inliers on the true linear model
 * y = TRUE_INTERCEPT + TRUE_SLOPE * x, evenly spaced from X_MIN to X_MAX.
 *
 * Params:
 *     points_x    array of n floats, modified in place
 *     points_y    array of n floats, modified in place
 *     n           int, number of inlier points
----------------------------------------------------------------------------- */
static void _make_line(float* points_x, float* points_y, int n) {
    float params[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    make_inliers(points_x, points_y, n, params, N_PARAMS, X_MIN, X_MAX);
}


/** ----------------------------------------------------------------------------
 * Adds zero mean gaussian noise with standard deviation std to
 * points_y in place. Thin wrapper around add_gaussian_noise.
 *
 * Params:
 *     points_y    array of n floats, modified in place
 *     n           int, number of points
 *     std         float, standard deviation of gaussian noise
 ---------------------------------------------------------------------------- */
static void _add_gaussian_noise(float* points_y, int n, float std) {
    add_gaussian_noise(points_y, n, std);
}


/** ----------------------------------------------------------------------------
 * Appends n_outliers = floor(n_inliers * epsilon) outlier points to
 * points_x and points_y in place. Outliers are guaranteed to lie
 * outside the inlier band defined by the true model and NOISE_STD.
 *
 * Params:
 *     points_x    array of floats, size >= n_inliers + n_outliers
 *     points_y    array of floats, size >= n_inliers + n_outliers
 *     n_inliers   int, number of existing inlier points
 *     epsilon     float, outlier fraction in (0, 1)
 ---------------------------------------------------------------------------- */
static void _add_outliers(float* points_x, float* points_y,
                           int n_inliers, float epsilon) {
    int n_outliers = (int)(n_inliers * epsilon); /* floor via int cast */
    float params[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    add_outliers(points_x, points_y, n_inliers, n_outliers,
                 params, N_PARAMS, NOISE_STD);
}


/* =============================================================================
 * Tests for compute_t which estimates the inlier threshold t as
 * mean + 2 * std of the vertical residuals from all points to the
 * preliminary least squares model.
 *
 * Happy paths:
 *     clean data no noise     t should be near 0.0
 *     gaussian noise std=0.5  mean residual ≈ 0, so t ≈ 2 * std
 *
 * Edge cases:
 *     n_points < 2            should return -1
 * ========================================================================== */
/* test threshold for clean data, should  be 0.0f */
void test_clean_data_t() {
	float points_x[N], points_y[N];
	float true_epsilon = 0.0f;
	float params[] = {TRUE_INTERCEPT, TRUE_SLOPE};
	_make_line(points_x, points_y, N);
	float t = compute_t(points_x, points_y, N, N_PARAMS);
	assert_almost_equal(t, 0, "clean_data");
}

/* 	test threshold for data with mean zero gaussian noise. 
	should be +/- 3 * NOISE_STD */
void test_gaussian_noise_data_t() {
	float points_x[N], points_y[N];
	float true_epsilon = 0.0f;
	float params[] = {TRUE_INTERCEPT, TRUE_SLOPE};
	_make_line(points_x, points_y, N);
	_add_gaussian_noise(points_y, N, NOISE_STD);
	float t = compute_t(points_x, points_y, N, N_PARAMS);
	//gaussian noise generated with box muller +/- 3 NOISE_STD
	assert_equal_delta(t, 0, 3 * NOISE_STD,"gaussian_noise_data");   
}

/* test for n_points < n_params, should return -1. */
void test_n_points_lt_n_params() {
	float points_x[1], points_y[1];
	float true_epsilon = 0.0f;
	float params[] = {TRUE_INTERCEPT, TRUE_SLOPE};
	_make_line(points_x, points_y, 1);
	float t = compute_t(points_x, points_y, 1, N_PARAMS);
	assert_almost_equal(t, -1, "n_points lt n_params");
}

/* =============================================================================
 * Tests for estimate_epsilon.
 *
 * estimate_epsilon is a rough first guess only. At low outlier fractions
 * the estimate may be reasonably close to the true value. At high outlier
 * fractions the preliminary least squares fit is corrupted, inflating
 * mean and std and causing the function to undercount outliers. Tests
 * reflect this limitation by using loose deltas at low fractions and
 * asserting only a valid range at high fractions.
 *
 * Happy paths:
 *     clean data no outliers  epsilon should be exactly 0.0
 *     outlier fraction 0.20   epsilon should be within 0.2 of true
 *     outlier fraction 0.40   epsilon should be within 0.4 of true
 *     outlier fraction 0.60   epsilon should be in valid range [0, 1)
 *                             accuracy not expected at this fraction
 *
 * Edge cases:
 *     n_points < 2            should return -1
 * ========================================================================== */

/* Clean data with no outliers. Estimated epsilon should be almost 0.0. */
void test_estimate_epsilon_clean_data() {
    float points_x[N], points_y[N];
    /* fill with inliers only */
    _make_line(points_x, points_y, N); 
    float epsilon = estimate_epsilon(points_x, points_y, N, N_PARAMS);
    assert_almost_equal(epsilon, 0.0f, "estimate_epsilon_clean_data");
}

/* Clean data with no outliers. Estimated epsilon should be almost 0.0. */
void test_estimate_epsilon_clean_quadratic() {
    float points_x[N], points_y[N];
    /* fill with inliers only */
    float a0 = 1.0f;
    float a1 = 1.0f;
    float a2 = 1.0f;
    float params[] = {a0, a1, a2};
    // TO DO FINISH THIS TEST
    float epsilon = estimate_epsilon(points_x, points_y, N, N_PARAMS);
    assert_almost_equal(epsilon, 0.0f, "estimate_epsilon_clean_quadratic");
}

/* 	20 percent outliers. estimate_epsilon is a rough guess only.
	Asserts epsilon is within delta = true_epsilon of true_epsilon. */
void test_estimate_epsilon_low_outlier_fraction() {
    float true_epsilon = 0.2f;
    /* 20 outliers */
    int n_outliers = (int)(N * true_epsilon);
    /* 120 points total */
    int n_total    = N + n_outliers;         

    /* pre-allocate enough space for inliers + outliers */
    float points_x[n_total], points_y[n_total];
	/* fill inliers */
    _make_line(points_x, points_y, N);
    /* append outliers */
    _add_outliers(points_x, points_y, N, true_epsilon);

    float epsilon = estimate_epsilon(points_x, points_y, n_total, N_PARAMS);
    assert_equal_delta(epsilon, true_epsilon, true_epsilon,
        "estimate_epsilon_low_outlier_fraction");
}


/* 	40 percent outliers. estimate_epsilon is a rough guess only.
	Asserts epsilon is within delta = true_epsilon of true_epsilon. */
void test_estimate_epsilon_medium_outlier_fraction() {
    float true_epsilon = 0.4f;
    /* 40 outliers */
    int n_outliers = (int)(N * true_epsilon);
    /* 140 points total */
    int n_total    = N + n_outliers;

    float points_x[n_total], points_y[n_total];

    _make_line(points_x, points_y, N);
    _add_outliers(points_x, points_y, N, true_epsilon);

    float epsilon = estimate_epsilon(points_x, points_y, n_total, N_PARAMS);
    assert_equal_delta(epsilon, true_epsilon, true_epsilon,
        "estimate_epsilon_medium_outlier_fraction");
}


/* 	60 percent outliers. Least squares preliminary fit is heavily corrupted at 
	this fraction. Asserts only that the returned value is a valid epsilon in 
	[0, 1) — accuracy not expected. */
void test_estimate_epsilon_high_outlier_fraction() {
    float true_epsilon = 0.6f;
    /* 60 outliers */
    int n_outliers = (int)(N * true_epsilon);
    /* 160 points total */
    int n_total    = N + n_outliers;

    float points_x[n_total], points_y[n_total];

    _make_line(points_x, points_y, N);
    _add_outliers(points_x, points_y, N, true_epsilon);

    float epsilon = estimate_epsilon(points_x, points_y, n_total, N_PARAMS);

    /* delta = 1.0 accepts any value in [0, 1) relative to true_epsilon */
    assert_equal_delta(epsilon, true_epsilon, 1.0f,
        "estimate_epsilon/high_outlier_fraction");
}


/* n_points = 1, should return -1. */
void test_estimate_epsilon_n_points_lt_n_params() {
    float points_x[1], points_y[1];
    /* single point, invalid input */
    _make_line(points_x, points_y, 1); 
    float epsilon = estimate_epsilon(points_x, points_y, 1, N_PARAMS);
    assert_almost_equal(epsilon, -1.0f,
        "estimate_epsilon/n_points_less_than_2");
}


/* =============================================================================
 * Tests for compute_k which computes the required number of RANSAC
 * iterations using the formula:
 *
 *     k = ceil(log(p) / log(1 - (1 - epsilon)^n_params))
 *
 * with default failure probability p = 0.01.
 *
 * Happy paths:
 *     epsilon = 0.10, n_params = 2    k should equal 3
 *     epsilon = 0.30, n_params = 2    k should equal 7
 *     epsilon = 0.50, n_params = 2    k should equal 17
 *     epsilon = 0.70, n_params = 2    k should equal 49
 *     failure_prob = 0.05             k should be less than at p = 0.01
 *
 * Edge cases:
 *     epsilon <= 0        should return -1
 *     epsilon >= 1        should return -1
 *     n_params < 2        should return -1
 *     failure_prob <= 0   should return -1
 *     failure_prob >= 1   should return -1
 * ========================================================================== */

/* for epsilon 0.1, n_params = 2, fail_pr = 0.01, should return 3 */
void test_epsilon_10pc() {
	float epsilon = 0.1;
	int n_params = 2;
	int k = compute_k(epsilon, n_params, FAIL_PR);
	assert_equal_int(k, 3, "epsilon 10%");
}

/* for epsilon 0.3, n_params = 2, fail_pr = 0.01, should return 7 */
void test_epsilon_30pc() {
	float epsilon = 0.3;
	int k = compute_k(epsilon, N_PARAMS, FAIL_PR);
	assert_equal_int(k, 7, "epsilon 30%");
}

/* for epsilon 0.5, n_params = 2, fail_pr = 0.01, should return 17 */
void test_epsilon_50pc() {
	float epsilon = 0.5;
	int k = compute_k(epsilon, N_PARAMS, FAIL_PR);
	assert_equal_int(k, 17, "epsilon 50%");
}

/* for epsilon 0.7, n_params = 2, fail_pr = 0.01, should return 49 */
void test_epsilon_70pc() {
	float epsilon = 0.7;
	int k = compute_k(epsilon, N_PARAMS, FAIL_PR);
	assert_equal_int(k, 49, "epsilon 70%");
}

/* for epsilon 0.9, n_params = 2, fail_pr = 0.01, should return 459 */
void test_epsilon_90pc() {
	float epsilon = 0.9;
	int k = compute_k(epsilon, N_PARAMS, FAIL_PR);
	assert_equal_int(k, 459, "epsilon 90%");
}

/* for epsilon 0.5, n_params = 3, fail_pr = 0.01, should return 35 */
void test_epsilon_50pc_n_params_3() {
	float epsilon = 0.5;
	int n_params = 3;
	int k = compute_k(epsilon, n_params, FAIL_PR);
	assert_equal_int(k, 35, "epsilon 50%, n_params 3");
}

/* for epsilon 0.5, n_params = 2, fail_pr = 0.01, should return 14 */
void test_epsilon_50pc_fail_pr_2pc() {
	float epsilon = 0.5;
	float fail_pr = 0.02;
	int k = compute_k(epsilon, N_PARAMS, fail_pr);
	// less that 17 for less strict model of fail_pr 0.01
	assert_equal_int(k, 14, "epsilon 50%, fail_pr = 0.02");  
}

/* for epsilon 0, should return error k = 1 */
void test_epsilon_0() {
	float epsilon = 0;
	int k = compute_k(epsilon, N_PARAMS, FAIL_PR);
	assert_equal_int(k, 1, "epsilon 0");
}

/* for epsilon 1, should return error -1 */
void test_epsilon_1() {
	float epsilon = 1;
	int k = compute_k(epsilon, N_PARAMS, FAIL_PR);
	assert_equal_int(k, -1, "epsilon 1");
}

/* n_params = 1, should return error -1 */
void test_n_params_less_than_2() {
	float epsilon = 0.1;
	int n_params = 1;
	int k = compute_k(epsilon, n_params, FAIL_PR);
	assert_equal_int(k, -1, "n_params < 2");
}

/* fail_pr = 0, should return -1 error */
void test_epsilon_fail_pr_0() {
	float epsilon = 0.1;
	float fail_pr = 0.0;
	int k = compute_k(epsilon, N_PARAMS, fail_pr);
	assert_equal_int(k, -1, "fail_pr = 0");  
}

/* fail_pr = 1, should return -1 error */
void test_epsilon_fail_pr_1() {
	float epsilon = 0.1;
	float fail_pr = 1.0;
	int k = compute_k(epsilon, N_PARAMS, fail_pr);
	assert_equal_int(k, -1, "fail_pr = 1");  
}


/* =============================================================================
 * Tests for compute_d which computes the expected inlier count as
 * floor((1 - epsilon) * n_points).
 *
 * Happy paths:
 *     epsilon = 0.20, n_points = 100    d should equal 80
 *     epsilon = 0.40, n_points = 100    d should equal 60
 *     epsilon = 0.50, n_points = 100    d should equal 50
 *
 * Edge cases:
 *     epsilon <= 0    should return -1
 *     epsilon >= 1    should return -1
 *     n_points < 2    should return -1
 * ========================================================================== */
/* epsilon = 0.20, n_points = 100. d should equal 80. */
void test_epsilon_20pc_d() {
	float epsilon = 0.20;
    int d = compute_d(epsilon, N);
    assert_equal_int(d, 80, "epsilon_20pc");
}

/* epsilon = 0.40, n_points = 100. d should equal 60. */
void test_epsilon_40pc_d() {
	float epsilon = 0.40;
    int d = compute_d(epsilon, N);
    assert_equal_int(d, 60, "epsilon_40pc");
}

/* epsilon = 0.50, n_points = 200. d should equal 100. */
void test_epsilon_50pc_n200() {
	float epsilon = 0.50;
	int n = 200;
    int d = compute_d(epsilon, n);
    assert_equal_int(d, 100, "epsilon_50pc_n200");
}

/* epsilon = 0.0, d should return error -1. */
void test_epsilon_0_d() {
	float epsilon = 0.0;
    int d = compute_d(epsilon, N);
    assert_equal_int(d, -1, "epsilon_0");
}

/* epsilon = 1.0, d should return error -1. */
void test_epsilon_1_d() {
	float epsilon = 1.0;
    int d = compute_d(epsilon, N);
    assert_equal_int(d, -1, "epsilon_1");
}

/* n_points less than 2, should return -1. */
void test_n_points_less_than_2_d() {
	float epsilon = .1;
	int n = 1;
    int d = compute_d(epsilon, n);
    assert_equal_int(d, -1, "n_points < 2");
}

/* =============================================================================
 * Tests fisher_yates draws m distinct indices from [0, N). Asserts:
 * 1. At least one value is not in its original position (something moved)
 * 2. All values 0 .. N-1 are present exactly once (valid permutation)
============================================================================= */

/* Test over all the array */
void test_fisher_yates() {
	int idx[N];
	int m = 2;
	fisher_yates(idx, N, m);

	// something moved
	int something_moved = 0; // boolean-like false
	for(int i = 0; i < N; i++){
		if(idx[i] != i) {
			something_moved = 1;
			break;
		}
	}

	int seen[N];
	for (int i = 0; i < N; i++)
    	seen[i] = 0;
    // pass idx once
	for (int i = 0; i < N; i++)
    	seen[idx[i]] = 1;
    // if every index is not in seen this will turn to 0
	int every_index_present = 1;
	for (int i = 0; i < N; i++) {
    	if (seen[i] == 0){
    	    every_index_present = 0;
    	}
	}

	assert_equal_int(something_moved, 1, "fisher yates/something_moved");
	assert_equal_int(every_index_present, 1, 
		"fisher yates/every_index_present");
}

/* Test over all first m in the array */
void test_fisher_yates_first_m_distinct() {
    int idx[N];
    int m = 3;
    fisher_yates(idx, N, m);
    /* check all first m values are in range [0, N) */
    int in_range = 1;
    for (int i = 0; i < m; i++)
        if (idx[i] < 0 || idx[i] >= N)
            in_range = 0;
    /* check first m values are all distinct */
    int distinct = 1;
    for (int i = 0; i < m; i++)
        for (int j = i + 1; j < m; j++)
            if (idx[i] == idx[j])
                distinct = 0;
    assert_equal_int(in_range, 1, "fisher_yates/first_m_in_range");
    assert_equal_int(distinct, 1, "fisher_yates/first_m_distinct");
}


/* =============================================================================
 * Tests for ransac which finds the best fitting polynomial model of
 * degree n_params - 1 from noisy data containing outliers using Random
 * Sample Consensus. It randomly samples n_params points, fits a model
 * via fit_model, counts inliers within threshold vertical residual
 * using find_model_inliers, and repeats k_resample times or until
 * expected_inliers are found. Results are stored in return_array.
 *
 * return_array layout:
 *     return_array[0]               number of inliers in best model
 *     return_array[1]               number of iterations actually run
 *     return_array[2..2+n_params-1] best model params (a0, a1, ...)
 *
 * Happy paths:
 *     clean data no outliers          recovers true model exactly
 *     low outlier fraction 20%        recovers true model within delta
 *     medium outlier fraction 40%     recovers true model within delta
 *     high outlier fraction 60%       documents RANSAC limits
 *     quadratic model n_params = 3    recovers a0, a1, a2 within delta
 *     early stop triggered            iterations run <= k
 *
 * Edge cases:
 *     n_points < 2                    return -1
 *     n_params < 2                    return -1
 *     k_resample < 1                  return -1
 *     threshold <= 0                  return -1
 *     expected_inliers > n_points     return -1
 *     n_points < n_params             return -1
 * ========================================================================== */

/* =============================================================================
 * HELPERS
 * ========================================================================== */


/* ---------------------------------------------------------------
Helper: generates noisy polynomial inlier data and appends outliers in place. 
Points follow y = a0 + a1*x + ... with gaussian noise.
Computes threshold t and expected inliers d from the data.

Params:
	points_x        pointer to array of floats, size >= n_inliers + n_outliers
	points_y        pointer to array of floats, size >= n_inliers + n_outliers
	n_inliers       int, number of inlier points
	n_outliers      int, number of outlier points
	params          float array of n_params coefficients [a0, a1, ...]
	n_params        int, number of model parameters
	noise_std       float, gaussian noise std, pass 0 for clean data
	t_out           float pointer, estimated threshold written here
	d_out           int pointer, expected inlier count written here
--------------------------------------------------------------- */
static void _make_data(float* points_x, float* points_y,
                        int n_inliers, int n_outliers,
                        float* params, int n_params, float noise_std,
                        float* t_out, int* d_out) {
    /* generate inlier points on the polynomial model */
    make_inliers(points_x, points_y, n_inliers, params, n_params,
                 X_MIN, X_MAX);

    /* add gaussian noise if requested */
    if (noise_std > 0.0f)
        add_gaussian_noise(points_y, n_inliers, noise_std);

    /* estimate threshold from noisy inlier data before adding outliers */
    float t = compute_t(points_x, points_y, n_inliers, n_params);
    // return a minimum threshold
    if (t == 0.0f) {
        t = 1e-4f;
     }
    *t_out = t;

    /* append outliers after threshold is estimated */
    if (noise_std > 0.0f && n_outliers > 0)
        add_outliers(points_x, points_y, n_inliers, n_outliers,
                     params, n_params, noise_std);

    /* compute expected inlier count d from epsilon */
    int n_points = n_inliers + n_outliers;
    float epsilon = (float) n_outliers / n_points;
    *d_out = (epsilon == 0.0f) ? n_points
                               : compute_d(epsilon, n_points);
}


/*
 * Clean data with no noise and no outliers.
 * k = 1, single iteration is enough for clean data.
 * Should recover true intercept and slope exactly.
 */
void test_ransac_clean_data_no_outliers() {
    int n_inliers = 100, n_outliers = 0;
    int n_points  = n_inliers + n_outliers;
    float params_true[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    float points_x[n_points], points_y[n_points];
    float t; int d;
    _make_data(points_x, points_y, n_inliers, n_outliers,
               params_true, N_PARAMS, 0.0f, &t, &d);
    /* k = 1 — single iteration sufficient for clean data */
    int k = 1;
    float return_array[2 + N_PARAMS];
    int ret = ransac(points_x, points_y, n_points, N_PARAMS, k, t, d, return_array);
    assert_equal_int(ret, 0, "clean_data/result");
    assert_almost_equal(return_array[2], TRUE_INTERCEPT,
    	"clean_data/intercept");
    assert_almost_equal(return_array[3], TRUE_SLOPE, "clean_data/slope");
}

/* 	80 inliers with gaussian noise, 20 outliers (20 percent).
	Should recover true intercept and slope within 2.5 * noise_std. */
void test_ransac_low_outlier_fraction() {
    int n_inliers = 80, n_outliers = 20;
    int n_points  = n_inliers + n_outliers;
    float epsilon = (float) n_outliers / n_points;
    float params_true[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    float points_x[n_points], points_y[n_points];
    float t; int d;
    _make_data(points_x, points_y, n_inliers, n_outliers,
               params_true, N_PARAMS, NOISE_STD, &t, &d);
    int k = compute_k(epsilon, N_PARAMS, 0.01f);
    float return_array[2 + N_PARAMS];
    ransac(points_x, points_y, n_points, N_PARAMS, k, t, d, return_array);
    assert_equal_delta(return_array[2], TRUE_INTERCEPT, 2.5f * NOISE_STD,
        "ransac/low_outlier/intercept");
    assert_equal_delta(return_array[3], TRUE_SLOPE, 2.5f * NOISE_STD,
        "ransac/low_outlier/slope");
}

/*	60 inliers with gaussian noise, 40 outliers (40 percent).
	Should recover true intercept and slope within 2.5 * noise_std. */
void test_ransac_medium_outlier_fraction() {
    int n_inliers = 60, n_outliers = 40;
    int n_points  = n_inliers + n_outliers;
    float epsilon = (float) n_outliers / n_points;
    float params_true[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    float points_x[n_points], points_y[n_points];
    float t; int d;
    _make_data(points_x, points_y, n_inliers, n_outliers,
               params_true, N_PARAMS, NOISE_STD, &t, &d);
    int k = compute_k(epsilon, N_PARAMS, 0.01f);
    float return_array[2 + N_PARAMS];
    ransac(points_x, points_y, n_points, N_PARAMS, k, t, d, return_array);
    assert_equal_delta(return_array[2], TRUE_INTERCEPT, 2.5f * NOISE_STD,
        "ransac/medium_outlier/intercept");
    assert_equal_delta(return_array[3], TRUE_SLOPE, 2.5f * NOISE_STD,
        "ransac/medium_outlier/slope");
}

/* 	40 inliers with gaussian noise, 60 outliers (60 percent). */
void test_ransac_high_outlier_fraction() {
    int n_inliers = 40, n_outliers = 60;
    int n_points  = n_inliers + n_outliers;
    float epsilon = (float) n_outliers / n_points;
    float params_true[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    float points_x[n_points], points_y[n_points];
    float t; int d;
    _make_data(points_x, points_y, n_inliers, n_outliers,
               params_true, N_PARAMS, NOISE_STD, &t, &d);
    int k = compute_k(epsilon, N_PARAMS, 0.01f);
    float return_array[2 + N_PARAMS];
    ransac(points_x, points_y, n_points, N_PARAMS, k, t, d, return_array);
    assert_equal_delta(return_array[2], TRUE_INTERCEPT, 2.5f * NOISE_STD,
        "ransac/high_outlier/intercept");
    assert_equal_delta(return_array[3], TRUE_SLOPE, 2.5f * NOISE_STD,
        "ransac/high_outlier/slope");
}

/* 	Quadratic model y = 1 + 1*x + 1*x^2, n_params = 3.
	20 percent outliers. Should recover a0, a1, a2 within delta.
	Verifies return_array indexing for n_params > 2:
	return_array[2] = a0
	return_array[3] = a1
	return_array[4] = a2 */
void test_ransac_n_params_3() {
    int n_params  = 3;
    int n_inliers = 80, n_outliers = 20;
    int n_points  = n_inliers + n_outliers;
    float epsilon = (float) n_outliers / n_points;
    float params_true[] = {1.0f, 1.0f, 1.0f};
    float points_x[n_points], points_y[n_points];
    float t; int d;
    _make_data(points_x, points_y, n_inliers, n_outliers,
               params_true, n_params, NOISE_STD, &t, &d);
    int k = compute_k(epsilon, n_params, 0.01f);
    float return_array[2 + n_params];
    ransac(points_x, points_y, n_points, n_params, k, t, d, return_array);
    assert_equal_delta(return_array[2], 1.0f, 2.0f * NOISE_STD,
        "ransac/n_params_3/a0");
    assert_equal_delta(return_array[3], 1.0f, 2.0f * NOISE_STD,
        "ransac/n_params_3/a1");
    assert_equal_delta(return_array[4], 1.0f, 2.0f * NOISE_STD,
        "ransac/n_params_3/a2");
}

/* expected_inliers set to 40 to trigger early stop.
 	Iterations actually run should be <= k.
 	Inliers found should be >= expected_inliers. */
void test_ransac_early_stop() {
    int n_inliers = 90, n_outliers = 10;
    int n_points  = n_inliers + n_outliers;
    float epsilon = (float) n_outliers / n_points;
    float params_true[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    float points_x[n_points], points_y[n_points];
    float t; int d;
    _make_data(points_x, points_y, n_inliers, n_outliers,
               params_true, N_PARAMS, NOISE_STD, &t, &d);
    d = 40; /* overwrite to trigger early stop */
    int k = compute_k(epsilon, N_PARAMS, 0.01f);
    float return_array[2 + N_PARAMS];
    ransac(points_x, points_y, n_points, N_PARAMS, k, t, d, return_array);
    /* inliers found should meet or exceed early stop threshold */
    assert_equal_int(return_array[0] >= d, 1,
        "ransac/early_stop/inliers_ge_d");
    /* iterations run should not exceed k */
    assert_equal_int(return_array[1] <= k, 1,
        "ransac/early_stop/iterations_le_k");
}

/* n_points = 1, should return -1. */
void test_ransac_n_points_less_than_2() {
    float points_x[1] = {0.0f}, points_y[1] = {0.0f};
    float return_array[2 + N_PARAMS];
    int ret = ransac(points_x, points_y, 1, N_PARAMS,
                     10, 1e-6f, 1, return_array);
    assert_equal_int(ret, -1, "ransac/n_points_lt_2");
}


/* n_params = 1, should return -1. */
void test_ransac_n_params_less_than_2() {
    float points_x[N], points_y[N];
    float return_array[2 + N_PARAMS];
    int ret = ransac(points_x, points_y, N, 1,
                     10, 1e-6f, 20, return_array);
    assert_equal_int(ret, -1, "ransac/n_params_lt_2");
}


/* k_resample = 0, should return -1. */
void test_ransac_k_resample_less_than_1() {
    float points_x[N], points_y[N];
    float return_array[2 + N_PARAMS];
    int ret = ransac(points_x, points_y, N, N_PARAMS,
                     0, 1e-6f, 20, return_array);
    assert_equal_int(ret, -1, "ransac/k_resample_lt_1");
}


/* threshold = 0, should return -1. */
void test_ransac_threshold_zero() {
    float points_x[N], points_y[N];
    float return_array[2 + N_PARAMS];
    int ret = ransac(points_x, points_y, N, N_PARAMS,
                     10, 0.0f, 20, return_array);
    assert_equal_int(ret, -1, "ransac/threshold_zero");
}


/* threshold = -1, should return -1. */
void test_ransac_threshold_negative() {
    float points_x[N], points_y[N];
    float return_array[2 + N_PARAMS];
    int ret = ransac(points_x, points_y, N, N_PARAMS,
                     10, -1.0f, 20, return_array);
    assert_equal_int(ret, -1, "ransac/threshold_negative");
}


/* expected_inliers > n_points, should return -1. */
void test_ransac_expected_inliers_greater_than_n_points() {
    float points_x[N], points_y[N];
    float return_array[2 + N_PARAMS];
    /* d = N + 1 exceeds n_points */
    int ret = ransac(points_x, points_y, N, N_PARAMS,
                     10, 1e-6f, N + 1, return_array);
    assert_equal_int(ret, -1, "ransac/expected_inliers_gt_n_points");
}


/* n_points < n_params, should return -1. */
void test_ransac_n_points_less_than_n_params() {
    float points_x[N], points_y[N];
    float return_array[2 + N_PARAMS];
    /* n_params = 150 exceeds n_points = N = 100 */
    int ret = ransac(points_x, points_y, N, 150,
                     10, 1e-6f, 20, return_array);
    assert_equal_int(ret, -1, "ransac/n_points_lt_n_params");
}
/* =============================================================================
  MAIN 
============================================================================= */
int main() {
	srand((unsigned int) time(NULL)); /* seed once for all random tests */

	printf("***** RUNNING TESTS FOR COMPUTE_T *****\n");
	test_clean_data_t();
	test_gaussian_noise_data_t();
	test_n_points_lt_n_params();
	printf("***** RUNNING TESTS FOR ESTIMATE_EPSILON *****\n");
	test_estimate_epsilon_clean_data();
	test_estimate_epsilon_low_outlier_fraction();
	test_estimate_epsilon_medium_outlier_fraction();
	test_estimate_epsilon_n_points_lt_n_params();
	printf("***** RUNNING TESTS FOR COMPUTE_K *****\n");
	test_epsilon_10pc();
	test_epsilon_30pc();
	test_epsilon_50pc();
	test_epsilon_70pc();
	test_epsilon_90pc();
	test_epsilon_50pc_n_params_3();
	test_epsilon_50pc_fail_pr_2pc();
	test_epsilon_0();
	test_epsilon_1();
	test_n_params_less_than_2();
	test_epsilon_fail_pr_0();
	test_epsilon_fail_pr_1();
	printf("***** RUNNING TESTS FOR COMPUTE_D *****\n");
	test_epsilon_20pc_d();
	test_epsilon_40pc_d();
	test_epsilon_50pc_n200();
	test_epsilon_0_d();
	test_epsilon_1_d();
	test_n_points_less_than_2_d();
	printf("***** RUNNING TESTS FOR FISHER YATES *****\n");
	test_fisher_yates();
	test_fisher_yates_first_m_distinct();
	printf("***** RUNNING TESTS FOR RANSAC *****\n");
	test_ransac_clean_data_no_outliers();
	test_ransac_low_outlier_fraction();
	test_ransac_medium_outlier_fraction();
	test_ransac_high_outlier_fraction();
	test_ransac_n_params_3();
	test_ransac_early_stop();
	test_ransac_n_points_less_than_2();
    test_ransac_n_params_less_than_2();
    test_ransac_k_resample_less_than_1();
    test_ransac_threshold_zero();
    test_ransac_threshold_negative();
    test_ransac_expected_inliers_greater_than_n_points();
    test_ransac_n_points_less_than_n_params();
	return 0;
}


