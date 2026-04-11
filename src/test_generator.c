/**
 * Tests for generator.h functions:
 *     make_inliers
 *     add_gaussian_noise
 *     add_structural_bias
 *     add_outliers
 *
 * make_inliers generates inlier points on a polynomial model of degree
 * n_params - 1, defined by params from lowest to highest degree [a0, a1, ...].
 *
 * add_outliers appends outlier points guaranteed to lie outside the inlier
 * band defined by the true polynomial model and noise_std, using vertical
 * residual consistent with find_model_inliers.
 */

#include "generator.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* constants for make_inliers and noise tests */
#define N        10
#define X_MIN    0.0f
#define X_MAX    9.0f
#define N_PARAMS 2
#define STD      2.0f

/* constants for outlier and bias tests */
#define N_INLIERS  90
#define N_OUTLIERS 10
#define N_POINTS   100
#define PR         1.0f

/* ================================================================
 * ASSERT HELPERS
 * ================================================================ */

static void assert_equal_int(int a, int b, const char* label) {
    if (a != b) printf("FAIL %s: expected %d got %d\n", label, b, a);
    else        printf("PASS %s\n", label);
}

/*
 * Asserts all elements of a equal corresponding elements of b within
 * 1e-5. Reports the first failing index and returns immediately.
 */
static void assert_array_almost_equal(float* a, float* b, int n,
                                       const char* label) {
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > 1e-5f) {
            printf("FAIL %s: index %d expected %.6f got %.6f\n",
                   label, i, b[i], a[i]);
            return;
        }
    }
    printf("PASS %s\n", label);
}

/*
 * Asserts all elements of a are greater than threshold.
 * Reports the first failing index and returns immediately.
 */
static void assert_array_gt(float* a, int n, float threshold,
                             const char* label) {
    for (int i = 0; i < n; i++) {
        if (a[i] <= threshold) {
            printf("FAIL %s: index %d value %.6f not > %.6f\n",
                   label, i, a[i], threshold);
            return;
        }
    }
    printf("PASS %s\n", label);
}


/* ================================================================
 * DATA HELPERS
 * ================================================================ */

/*
 * Fills points_x and points_y with n inliers on the polynomial defined
 * by params and n_params, over [X_MIN, X_MAX].
 */
static void _make(float* points_x, float* points_y, int n,
                  float* params, int n_params) {
    make_inliers(points_x, points_y, n, params, n_params, X_MIN, X_MAX);
}

/*
 * Asserts points_x[i] == i and points_y[i] == intercept + slope*i
 * for all i in [0, n) using array comparison.
 */
static void _assert_line(float* points_x, float* points_y, int n,
                          float intercept, float slope, const char* tag) {
    float exp_x[n], exp_y[n];
    for (int i = 0; i < n; i++) {
        exp_x[i] = (float) i;
        exp_y[i] = intercept + slope * (float) i;
    }
    char label[128];
    snprintf(label, sizeof(label), "%s/points_x", tag);
    assert_array_almost_equal(points_x, exp_x, n, label);
    snprintf(label, sizeof(label), "%s/points_y", tag);
    assert_array_almost_equal(points_y, exp_y, n, label);
}

/*
 * Asserts all points_y[i] changed from clean and remain within tol
 * of the clean value. Uses array diff against clean and copy arrays.
 */
static void _assert_noise(float* points_y, float* clean_y, int n,
                           float tol, const char* tag) {
    /* build array of absolute differences from clean */
    float diffs[n];
    for (int i = 0; i < n; i++)
        diffs[i] = fabsf(points_y[i] - clean_y[i]);

    /* all diffs should be > 0 (noise was added) */
    char label[128];
    snprintf(label, sizeof(label), "%s/noise_added", tag);
    assert_array_gt(diffs, n, 1e-6f, label);

    /* all diffs should be < tol */
    float tol_arr[n];
    for (int i = 0; i < n; i++)
        tol_arr[i] = tol - diffs[i]; /* positive means within tol */
    snprintf(label, sizeof(label), "%s/within_tol", tag);
    assert_array_gt(tol_arr, n, 0.0f, label);
}

/*
 * Asserts inliers points_y[0..n_inliers-1] match copy_y, and outliers
 * points_y[n_inliers..n_total-1] lie outside band 2*std from model.
 */
static void _assert_outliers(float* points_x, float* points_y,
                              float* copy_y, int n_inliers, int n_total,
                              float intercept, float slope, float std,
                              const char* tag) {
    /* check inliers unchanged */
    char label[128];
    snprintf(label, sizeof(label), "%s/inliers_unchanged", tag);
    assert_array_almost_equal(points_y, copy_y, n_inliers, label);

    /* check outliers outside band */
    int n_outliers = n_total - n_inliers;
    float dists[n_outliers];
    for (int i = 0; i < n_outliers; i++) {
        int idx = n_inliers + i;
        dists[i] = fabsf(points_y[idx]
                         - (intercept + slope * points_x[idx]));
    }
    snprintf(label, sizeof(label), "%s/outliers_outside_band", tag);
    assert_array_gt(dists, n_outliers, 2.0f * std, label);
}

/*
 * Asserts points_y[i] == copy_y[i] + bias_fn(points_x[i])
 * for all i in [0, N_POINTS) using array comparison.
 */
static void _assert_bias(float* points_x, float* points_y, float* copy_y,
                          float (*bias_fn)(float), const char* tag) {
    float expected[N_POINTS];
    for (int i = 0; i < N_POINTS; i++)
        expected[i] = copy_y[i] + bias_fn(points_x[i]);
    assert_array_almost_equal(points_y, expected, N_POINTS, tag);
}

/*
 * Fills points_x and points_y with N_POINTS inliers on y = x,
 * copies points_y into copy_y.
 */
static void _setup_bias(float* points_x, float* points_y, float* copy_y) {
    float params[] = {0.0f, 1.0f};
    _make(points_x, points_y, N_POINTS, params, N_PARAMS);
    memcpy(copy_y, points_y, N_POINTS * sizeof(float));
}

/* bias functions used as arguments */
static float _bias_zero(float x)     { (void)x; return 0.0f; }
static float _bias_const(float x)    { (void)x; return 1.0f; }
static float _bias_linear(float x)   { return 0.5f * x; }
static float _bias_periodic(float x) { return sinf(x); }


/* =============================================================================
 * MAKE_INLIERS
 *
 * Happy paths:
 *     unit slope zero intercept       y = 0 + 1*x
 *     negative unit slope             y = N + (-1)*x
 *     float slope zero intercept      y = 0 + 0.5*x
 *     unit slope negative intercept   y = -0.5 + 1*x
 *     zero slope zero intercept       y = 0 (flat line)
 *     quadratic model                 y = 1 + 1*x + 1*x^2
 * Edge cases:
 *     n < 2               should return -1
 *     x_min == x_max      should return -1
 *     n_params < 2        should return -1
 ============================================================================ */

void test_make_inliers_slope_1_intercept_0() {
    float points_x[N], points_y[N];
    float params[] = {0.0f, 1.0f};
    _make(points_x, points_y, N, params, N_PARAMS);
    _assert_line(points_x, points_y, N, 0.0f, 1.0f,
                 "make_inliers/slope=1/intercept=0");
}

void test_make_inliers_slope_neg1_intercept_n() {
    float points_x[N], points_y[N];
    float params[] = {(float)N, -1.0f};
    _make(points_x, points_y, N, params, N_PARAMS);
    _assert_line(points_x, points_y, N, (float)N, -1.0f,
                 "make_inliers/slope=-1/intercept=N");
}

void test_make_inliers_slope_float_intercept_0() {
    float points_x[N], points_y[N];
    float params[] = {0.0f, 0.5f};
    _make(points_x, points_y, N, params, N_PARAMS);
    _assert_line(points_x, points_y, N, 0.0f, 0.5f,
                 "make_inliers/slope=0.5/intercept=0");
}

void test_make_inliers_slope_1_intercept_neg_float() {
    float points_x[N], points_y[N];
    float params[] = {-0.5f, 1.0f};
    _make(points_x, points_y, N, params, N_PARAMS);
    _assert_line(points_x, points_y, N, -0.5f, 1.0f,
                 "make_inliers/slope=1/intercept=-0.5");
}

void test_make_inliers_slope_0_intercept_0() {
    float points_x[N], points_y[N];
    float params[] = {0.0f, 0.0f};
    _make(points_x, points_y, N, params, N_PARAMS);
    _assert_line(points_x, points_y, N, 0.0f, 0.0f,
                 "make_inliers/slope=0/intercept=0");
}

/* y = 1 + 1*x + 1*x^2 */
void test_make_inliers_quadratic_model() {
    float points_x[N], points_y[N];
    float params[] = {1.0f, 1.0f, 1.0f};
    _make(points_x, points_y, N, params, 3);
    float exp_x[N], exp_y[N];
    for (int i = 0; i < N; i++) {
        exp_x[i] = (float) i;
        exp_y[i] = 1.0f + (float)i + (float)(i * i);
    }
    assert_array_almost_equal(points_x, exp_x, N,
        "make_inliers/quadratic/points_x");
    assert_array_almost_equal(points_y, exp_y, N,
        "make_inliers/quadratic/points_y");
}

void test_make_inliers_edge_n_lt_2() {
    float points_x[1], points_y[1];
    float params[] = {0.0f, 1.0f};
    assert_equal_int(
        make_inliers(points_x, points_y, 1, params, N_PARAMS, X_MIN, X_MAX),
        -1, "make_inliers/n<2");
}

void test_make_inliers_edge_xmin_eq_xmax() {
    float points_x[N], points_y[N];
    float params[] = {0.0f, 1.0f};
    assert_equal_int(
        make_inliers(points_x, points_y, N, params, N_PARAMS, 5.0f, 5.0f),
        -1, "make_inliers/xmin==xmax");
}

void test_make_inliers_edge_n_params_lt_2() {
    float points_x[N], points_y[N];
    float params[] = {0.0f};
    assert_equal_int(
        make_inliers(points_x, points_y, N, params, 1, X_MIN, X_MAX),
        -1, "make_inliers/n_params<2");
}


/* =============================================================================
 * ADD_GAUSSIAN_NOISE
 *
 * Happy paths:
 *     std int     all points_y[i] changed and within 3*std of clean
 *     std float   all points_y[i] changed and within 3*std of clean
 * Edge cases:
 *     n_inliers < 2   should return -1
 *     std == 0        should return -1
 *     std < 0         should return -1
 ============================================================================ */

void test_add_gaussian_noise_std_int() {
    float points_x[N], points_y[N], clean_y[N];
    float params[] = {0.0f, 1.0f};
    _make(points_x, points_y, N, params, N_PARAMS);
    memcpy(clean_y, points_y, N * sizeof(float));
    add_gaussian_noise(points_y, N, STD);
    _assert_noise(points_y, clean_y, N, 3.0f * STD,
                  "add_gaussian_noise/std=2");
}

void test_add_gaussian_noise_std_float() {
    float points_x[N], points_y[N], clean_y[N];
    float std = 2.5f;
    float params[] = {0.0f, 1.0f};
    _make(points_x, points_y, N, params, N_PARAMS);
    memcpy(clean_y, points_y, N * sizeof(float));
    add_gaussian_noise(points_y, N, std);
    _assert_noise(points_y, clean_y, N, 3.0f * std,
                  "add_gaussian_noise/std=2.5");
}

void test_add_gaussian_noise_n_inliers_lt_2() {
    float points_y[N];
    assert_equal_int(add_gaussian_noise(points_y, 1, STD), -1,
                     "add_gaussian_noise/n<2");
}

void test_add_gaussian_noise_std_0() {
    float points_y[N];
    assert_equal_int(add_gaussian_noise(points_y, N, 0.0f), -1,
                     "add_gaussian_noise/std=0");
}

void test_add_gaussian_noise_std_lt_0() {
    float points_y[N];
    assert_equal_int(add_gaussian_noise(points_y, N, -2.0f), -1,
                     "add_gaussian_noise/std<0");
}


/* =============================================================================
 * ADD_OUTLIERS
 *
 * Happy paths:
 *     inlier points unchanged after call
 *     n_outliers points appended, all outside band |y - model(x)| > 2*std
 * Special cases:
 *     n_outliers == 0           data unchanged, return 0
 *     y_min == y_max (slope=0)  return 0, inliers unchanged, outliers outside
 * Edge cases:
 *     n_inliers < 0              should return -1
 *     n_outliers < 0             should return -1
 *     n_inliers + n_outliers < 2 should return -1
 *     noise_std <= 0             should return -1
 *     n_params < 2               should return -1
 *     x_min == x_max             should return -1
 ============================================================================ */

void test_add_outliers_happy_path() {
    float points_x[N_POINTS], points_y[N_POINTS], copy_y[N_INLIERS];
    float params[] = {0.0f, 1.0f};
    _make(points_x, points_y, N_INLIERS, params, N_PARAMS);
    add_gaussian_noise(points_y, N_INLIERS, STD);
    memcpy(copy_y, points_y, N_INLIERS * sizeof(float));
    add_outliers(points_x, points_y, N_INLIERS, N_OUTLIERS,
                 params, N_PARAMS, STD);
    _assert_outliers(points_x, points_y, copy_y, N_INLIERS, N_POINTS,
                     0.0f, 1.0f, STD, "add_outliers/happy_path");
}

void test_add_outliers_n_outliers_0() {
    float points_x[N_POINTS], points_y[N_POINTS];
    float params[] = {0.0f, 1.0f};
    _make(points_x, points_y, N_POINTS, params, N_PARAMS);
    assert_equal_int(
        add_outliers(points_x, points_y, N_POINTS, 0,
                     params, N_PARAMS, STD),
        0, "add_outliers/n_outliers=0");
}

void test_add_outliers_y_min_eq_y_max() {
    float points_x[N_POINTS], points_y[N_POINTS], copy_y[N_INLIERS];
    float params[] = {0.0f, 0.0f}; /* slope=0, y_min==y_max */
    _make(points_x, points_y, N_INLIERS, params, N_PARAMS);
    memcpy(copy_y, points_y, N_INLIERS * sizeof(float));
    assert_equal_int(
        add_outliers(points_x, points_y, N_INLIERS, N_OUTLIERS,
                     params, N_PARAMS, STD),
        0, "add_outliers/y_min==y_max/result");
    _assert_outliers(points_x, points_y, copy_y, N_INLIERS, N_POINTS,
                     0.0f, 0.0f, STD, "add_outliers/y_min==y_max");
}

void test_add_outliers_n_inliers_lt_0() {
    float points_x[N_INLIERS], points_y[N_INLIERS];
    float params[] = {0.0f, 1.0f};
    assert_equal_int(
        add_outliers(points_x, points_y, -1, N_OUTLIERS,
                     params, N_PARAMS, STD),
        -1, "add_outliers/n_inliers<0");
}

void test_add_outliers_n_outliers_lt_0() {
    float points_x[N_INLIERS], points_y[N_INLIERS];
    float params[] = {0.0f, 1.0f};
    assert_equal_int(
        add_outliers(points_x, points_y, N_INLIERS, -1,
                     params, N_PARAMS, STD),
        -1, "add_outliers/n_outliers<0");
}

void test_add_outliers_n_points_lt_2() {
    float points_x[N_INLIERS], points_y[N_INLIERS];
    float params[] = {0.0f, 1.0f};
    assert_equal_int(
        add_outliers(points_x, points_y, 0, 1,
                     params, N_PARAMS, STD),
        -1, "add_outliers/n_points<2");
}

void test_add_outliers_std_0() {
    float points_x[N_INLIERS], points_y[N_INLIERS];
    float params[] = {0.0f, 1.0f};
    assert_equal_int(
        add_outliers(points_x, points_y, N_INLIERS, N_OUTLIERS,
                     params, N_PARAMS, 0.0f),
        -1, "add_outliers/std=0");
}

void test_add_outliers_std_lt_0() {
    float points_x[N_INLIERS], points_y[N_INLIERS];
    float params[] = {0.0f, 1.0f};
    assert_equal_int(
        add_outliers(points_x, points_y, N_INLIERS, N_OUTLIERS,
                     params, N_PARAMS, -1.0f),
        -1, "add_outliers/std<0");
}

void test_add_outliers_n_params_lt_2() {
    float points_x[N_INLIERS], points_y[N_INLIERS];
    float params[] = {0.0f, 1.0f};
    assert_equal_int(
        add_outliers(points_x, points_y, N_INLIERS, N_OUTLIERS,
                     params, 1, STD),
        -1, "add_outliers/n_params<2");
}

void test_add_outliers_x_min_eq_x_max() {
    float points_x[N_INLIERS], points_y[N_INLIERS];
    float params[] = {0.0f, 1.0f};
    _make(points_x, points_y, N_INLIERS, params, N_PARAMS);
    for (int i = 0; i < N_INLIERS; i++)
        points_x[i] = X_MAX; /* force all x equal */
    assert_equal_int(
        add_outliers(points_x, points_y, N_INLIERS, N_OUTLIERS,
                     params, N_PARAMS, STD),
        -1, "add_outliers/x_min==x_max");
}


/* =============================================================================
 * ADD_STRUCTURAL_BIAS
 *
 * Happy paths:
 *     bias_fn = 0         points_y unchanged
 *     bias_fn = 1.0       points_y[i] increased by 1.0
 *     bias_fn = 0.5*x     points_y[i] increased by 0.5*x
 *     bias_fn = sin(x)    points_y[i] increased by sin(x)
 * Edge cases:
 *     bias_fn is NULL     should return -1
 *     pr < 0              should return -1
 *     pr > 1              should return -1
 *     n_inliers < 2       should return -1
 ============================================================================ */

void test_add_structural_bias_const_0() {
    float points_x[N_POINTS], points_y[N_POINTS], copy_y[N_POINTS];
    _setup_bias(points_x, points_y, copy_y);
    assert_equal_int(
        add_structural_bias(points_y, points_x, N_POINTS, PR, _bias_zero),
        0, "add_structural_bias/bias=0/result");
    _assert_bias(points_x, points_y, copy_y, _bias_zero,
                 "add_structural_bias/bias=0");
}

void test_add_structural_bias_const_1() {
    float points_x[N_POINTS], points_y[N_POINTS], copy_y[N_POINTS];
    _setup_bias(points_x, points_y, copy_y);
    assert_equal_int(
        add_structural_bias(points_y, points_x, N_POINTS, PR, _bias_const),
        0, "add_structural_bias/bias=1/result");
    _assert_bias(points_x, points_y, copy_y, _bias_const,
                 "add_structural_bias/bias=1");
}

void test_add_structural_bias_linear() {
    float points_x[N_POINTS], points_y[N_POINTS], copy_y[N_POINTS];
    _setup_bias(points_x, points_y, copy_y);
    assert_equal_int(
        add_structural_bias(points_y, points_x, N_POINTS, PR, _bias_linear),
        0, "add_structural_bias/bias=linear/result");
    _assert_bias(points_x, points_y, copy_y, _bias_linear,
                 "add_structural_bias/bias=linear");
}

void test_add_structural_bias_periodic() {
    float points_x[N_POINTS], points_y[N_POINTS], copy_y[N_POINTS];
    _setup_bias(points_x, points_y, copy_y);
    assert_equal_int(
        add_structural_bias(points_y, points_x, N_POINTS, PR, _bias_periodic),
        0, "add_structural_bias/bias=sin/result");
    _assert_bias(points_x, points_y, copy_y, _bias_periodic,
                 "add_structural_bias/bias=sin");
}

void test_add_structural_bias_null() {
    float points_x[N_POINTS], points_y[N_POINTS], copy_y[N_POINTS];
    _setup_bias(points_x, points_y, copy_y);
    assert_equal_int(
        add_structural_bias(points_y, points_x, N_POINTS, PR, NULL),
        -1, "add_structural_bias/NULL");
}

void test_add_structural_bias_pr_lt_0() {
    float points_x[N_POINTS], points_y[N_POINTS], copy_y[N_POINTS];
    _setup_bias(points_x, points_y, copy_y);
    assert_equal_int(
        add_structural_bias(points_y, points_x, N_POINTS, -1.0f, _bias_const),
        -1, "add_structural_bias/pr<0");
}

void test_add_structural_bias_pr_gt_1() {
    float points_x[N_POINTS], points_y[N_POINTS], copy_y[N_POINTS];
    _setup_bias(points_x, points_y, copy_y);
    assert_equal_int(
        add_structural_bias(points_y, points_x, N_POINTS, 1.1f, _bias_const),
        -1, "add_structural_bias/pr>1");
}

void test_add_structural_bias_n_inliers_1() {
    float points_x[N_POINTS], points_y[N_POINTS], copy_y[N_POINTS];
    _setup_bias(points_x, points_y, copy_y);
    assert_equal_int(
        add_structural_bias(points_y, points_x, 1, PR, _bias_const),
        -1, "add_structural_bias/n<2");
}


/* ================================================================
 * MAIN
 * ================================================================ */

int main() {
    srand((unsigned int) time(NULL));

    printf("***** MAKE_INLIERS *****\n");
    test_make_inliers_slope_1_intercept_0();
    test_make_inliers_slope_neg1_intercept_n();
    test_make_inliers_slope_float_intercept_0();
    test_make_inliers_slope_1_intercept_neg_float();
    test_make_inliers_slope_0_intercept_0();
    test_make_inliers_quadratic_model();
    test_make_inliers_edge_n_lt_2();
    test_make_inliers_edge_xmin_eq_xmax();
    test_make_inliers_edge_n_params_lt_2();

    printf("***** ADD_GAUSSIAN_NOISE *****\n");
    test_add_gaussian_noise_std_int();
    test_add_gaussian_noise_std_float();
    test_add_gaussian_noise_n_inliers_lt_2();
    test_add_gaussian_noise_std_0();
    test_add_gaussian_noise_std_lt_0();

    printf("***** ADD_OUTLIERS *****\n");
    test_add_outliers_happy_path();
    test_add_outliers_n_outliers_0();
    test_add_outliers_y_min_eq_y_max();
    test_add_outliers_n_inliers_lt_0();
    test_add_outliers_n_outliers_lt_0();
    test_add_outliers_n_points_lt_2();
    test_add_outliers_std_0();
    test_add_outliers_std_lt_0();
    test_add_outliers_n_params_lt_2();
    test_add_outliers_x_min_eq_x_max();

    printf("***** ADD_STRUCTURAL_BIAS *****\n");
    test_add_structural_bias_const_0();
    test_add_structural_bias_const_1();
    test_add_structural_bias_linear();
    test_add_structural_bias_periodic();
    test_add_structural_bias_null();
    test_add_structural_bias_pr_lt_0();
    test_add_structural_bias_pr_gt_1();
    test_add_structural_bias_n_inliers_1();

    return 0;
}