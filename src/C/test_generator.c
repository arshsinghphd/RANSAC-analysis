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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* for make_inliers, noise */
#define N        10
#define X_MIN    0.0
#define X_MAX    9.0
#define N_PARAMS 2
#define STD      2.0

/* additional for outlier testing */
#define N_INLIERS  90
#define N_OUTLIERS 10
#define N_POINTS   100

/* additional for structural bias testing */
#define PR 1

/* ================================================================
 * HELPERS
 * ================================================================ */

static void assert_almost_equal(double a, double b, const char *label) {
    if (fabs(a - b) > 1e-5)
        printf("FAIL %s: expected %.6f got %.6f\n", label, b, a);
    else
        printf("PASS %s\n", label);
}

static void assert_equal_int(int a, int b, const char *label) {
    if (a != b) printf("FAIL %s: expected %d got %d\n", label, b, a);
    else        printf("PASS %s\n", label);
}

/** Assert all points_x[i] == i and points_y[i] == intercept + slope*i. */
static void assert_line_points(float *points_x, float *points_y, int n,
                               float intercept, float slope,
                               const char *tag) {
    char label[128];
    for (int i = 0; i < n; i++) {
        snprintf(label, sizeof(label), "%s: points_x[%d]", tag, i);
        assert_almost_equal(points_x[i], i, label);
        snprintf(label, sizeof(label), "%s: points_y[%d]", tag, i);
        assert_almost_equal(points_y[i], intercept + slope * i, label);
    }
}

/** Assert all points_y[i] changed from clean and stay within tol of clean. */
static void assert_noise_added(float *points_y, int n,
                               float intercept, float slope,
                               double tol, const char *tag) {
    char label[128];
    for (int i = 0; i < n; i++) {
        double clean = intercept + slope * i;
        snprintf(label, sizeof(label), "%s: points_y[%d] unchanged", tag, i);
        assert_almost_equal(fabs(points_y[i] - clean) > 1e-6 ? 1.0 : 0.0, 1.0, label);
        snprintf(label, sizeof(label), "%s: points_y[%d] outside tol", tag, i);
        assert_almost_equal(fabs(points_y[i] - clean) < tol ? 1.0 : 0.0, 1.0, label);
    }
}

/* ================================================================
 * MAKE_INLIERS
 * ================================================================
 * Happoints_y paths:
 *   unit slope zero intercept       y = 0 + 1*x
 *   negative unit slope             y = N + (-1)*x
 *   float slope zero intercept      y = 0 + 0.5*x
 *   unit slope negative intercept   y = -0.5 + 1*x
 *   zero slope zero intercept       y = 0 (flat line)
 *   quadratic model                 y = 1 + 1*x + 1*x^2
 * Edge cases:
 *   n < 2               should return -1
 *   x_min == x_max      should return -1
 *   n_params < 2        should return -1
 */

/** Test happoints_y path: slope = 1, intercept = 0. y = 0 + 1*x */
void test_make_inliers_slope_1_intercept_0() {
    float points_x[N], points_y[N];
    float params[] = {0.0f, 1.0f};
    make_inliers(points_x, points_y, N, params, N_PARAMS, X_MIN, X_MAX);
    assert_line_points(points_x, points_y, N, 0.0f, 1.0f,
                       "make_inliers slope=1 intercept=0");
}

/** Test happoints_y path: slope = -1, intercept = N. y = N + (-1)*x */
void test_make_inliers_slope_neg1_intercept_n() {
    float points_x[N], points_y[N];
    float params[] = {(float)N, -1.0f};
    make_inliers(points_x, points_y, N, params, N_PARAMS, X_MIN, X_MAX);
    assert_line_points(points_x, points_y, N, (float)N, -1.0f,
                       "make_inliers slope=-1 intercept=N");
}

/** Test happoints_y path: slope = 0.5, intercept = 0. y = 0 + 0.5*x */
void test_make_inliers_slope_float_intercept_0() {
    float points_x[N], points_y[N];
    float params[] = {0.0f, 0.5f};
    make_inliers(points_x, points_y, N, params, N_PARAMS, X_MIN, X_MAX);
    assert_line_points(points_x, points_y, N, 0.0f, 0.5f,
                       "make_inliers slope=0.5 intercept=0");
}

/** Test happoints_y path: slope=1, intercept=-0.5. y = -0.5 + 1*x */
void test_make_inliers_slope_1_intercept_neg_float() {
    float points_x[N], points_y[N];
    float params[] = {-0.5f, 1.0f};
    make_inliers(points_x, points_y, N, params, N_PARAMS, X_MIN, X_MAX);
    assert_line_points(points_x, points_y, N, -0.5f, 1.0f,
                       "make_inliers slope=1 intercept=-0.5");
}

/** Test happoints_y path: slope=0, intercept=0. y = 0 (flat line) */
void test_make_inliers_slope_0_intercept_0() {
    float points_x[N], points_y[N];
    float params[] = {0.0f, 0.0f};
    make_inliers(points_x, points_y, N, params, N_PARAMS, X_MIN, X_MAX);
    assert_line_points(points_x, points_y, N, 0.0f, 0.0f,
                       "make_inliers slope=0 intercept=0");
}

/** Test happoints_y path: quadratic model. y = 1 + 1*x + 1*x^2 */
void test_make_inliers_quadratic_model() {
    float points_x[N], points_y[N];
    float a = 1.0f, b = 1.0f, c = 1.0f;
    float params[] = {a, b, c};
    make_inliers(points_x, points_y, N, params, 3, X_MIN, X_MAX);
    char label[128];
    for (int i = 0; i < N; i++) {
        snprintf(label, sizeof(label), "make_inliers quadratic points_x[%d]", i);
        assert_almost_equal(points_x[i], i, label);
        snprintf(label, sizeof(label), "make_inliers quadratic points_y[%d]", i);
        assert_almost_equal(points_y[i], a + b*i + c*i*i, label);
    }
}

/** Test edge case: n < 2, should return -1. */
void test_make_inliers_edge_n_lt_2() {
    float points_x[1], points_y[1];
    float params[] = {0.0f, 1.0f};
    int ret = make_inliers(points_x, points_y, 1, params, N_PARAMS, X_MIN, X_MAX);
    assert_equal_int(ret, -1, "make_inliers n<2 returns -1");
}

/** Test edge case: x_min == x_max, should return -1. */
void test_make_inliers_edge_xmin_eq_xmax() {
    float points_x[N], points_y[N];
    float params[] = {0.0f, 1.0f};
    int ret = make_inliers(points_x, points_y, N, params, N_PARAMS, 5.0f, 5.0f);
    assert_equal_int(ret, -1, "make_inliers xmin==xmax returns -1");
}

/** Test edge case: n_params < 2, should return -1. */
void test_make_inliers_edge_n_params_lt_2() {
    float points_x[N], points_y[N];
    float params[] = {0.0f};
    int ret = make_inliers(points_x, points_y, N, params, 1, X_MIN, X_MAX);
    assert_equal_int(ret, -1, "make_inliers n_params<2 returns -1");
}


/* ================================================================
 * ADD_GAUSSIAN_NOISE
 * ================================================================
 * Happoints_y paths: all noisy points_y[i] within 3*std of clean value
 *   std int
 *   std float
 * Edge cases:
 *   n_inliers < 2   should return -1
 *   std == 0        should return -1
 *   std < 0         should return -1
 */

/** Test happoints_y path: std=2 (int). points_y changed and within 3*std of clean. */
void test_add_gaussian_noise_std_int() {
    float points_x[N], points_y[N];
    float params[] = {0.0f, 1.0f};
    make_inliers(points_x, points_y, N, params, N_PARAMS, X_MIN, X_MAX);
    add_gaussian_noise(points_y, N, STD);
    assert_noise_added(points_y, N, 0.0f, 1.0f, 3.0 * STD,
                       "add_gaussian_noise std=2");
}

/** Test happoints_y path: std=2.5 (float). points_y changed and within 3*std of clean. */
void test_add_gaussian_noise_std_float() {
    float points_x[N], points_y[N];
    float std = 2.5f;
    float params[] = {0.0f, 1.0f};
    make_inliers(points_x, points_y, N, params, N_PARAMS, X_MIN, X_MAX);
    add_gaussian_noise(points_y, N, std);
    assert_noise_added(points_y, N, 0.0f, 1.0f, 3.0 * std,
                       "add_gaussian_noise std=2.5");
}

/** Test edge case: n_inliers < 2, should return -1. */
void test_add_gaussian_noise_n_inliers_lt_2() {
    float points_y[N];
    int ret = add_gaussian_noise(points_y, 1, STD);
    assert_equal_int(ret, -1, "add_gaussian_noise n<2 returns -1");
}

/** Test edge case: std == 0, should return -1. */
void test_add_gaussian_noise_std_0() {
    float points_y[N];
    int ret = add_gaussian_noise(points_y, N, 0.0f);
    assert_equal_int(ret, -1, "add_gaussian_noise std=0 returns -1");
}

/** Test edge case: std < 0, should return -1. */
void test_add_gaussian_noise_std_lt_0() {
    float points_y[N];
    int ret = add_gaussian_noise(points_y, N, -2.0f);
    assert_equal_int(ret, -1, "add_gaussian_noise std<0 returns -1");
}


/* ================================================================
 * ADD_OUTLIERS
 * ================================================================
 * Happoints_y paths:
 *   inlier points unchanged after call
 *   n_outliers points appended, all outside band |y - model(x)| > 2*std
 * Special cases:
 *   n_outliers == 0            data unchanged, return 0
 *   y_min == y_max (slope=0)   return 0; inliers unchanged; outliers outside band
 * Edge cases:
 *   n_inliers < 0              should return -1
 *   n_outliers < 0             should return -1
 *   n_inliers + n_outliers < 2 should return -1
 *   noise_std <= 0             should return -1
 *   n_params < 2               should return -1
 *   x_min == x_max             should return -1
 */

/** Test happoints_y path: inliers unchanged; outliers appended outside band. */
void test_add_outliers_happoints_y_path() {
    float points_x[N_POINTS], points_y[N_POINTS], copoints_y_y[N_INLIERS];
    float params[] = {0.0f, 1.0f};
    make_inliers(points_x, points_y, N_INLIERS, params, N_PARAMS, X_MIN, X_MAX);
    add_gaussian_noise(points_y, N_INLIERS, STD);
    memcpy(copoints_y_y, points_y, N_INLIERS * sizeof(float));
    add_outliers(points_x, points_y, N_INLIERS, N_OUTLIERS, params, N_PARAMS, STD);
    char label[128];
    for (int i = 0; i < N_POINTS; i++) {
        if (i < N_INLIERS) {
            snprintf(label, sizeof(label), "add_outliers happoints_y: inlier points_y[%d] changed", i);
            assert_almost_equal(points_y[i], copoints_y_y[i], label);
        } else {
            snprintf(label, sizeof(label), "add_outliers happoints_y: outlier points_y[%d] inside band", i);
            assert_almost_equal(fabs(points_y[i] - (0.0f + 1.0f * points_x[i])) > 2*STD ? 1.0 : 0.0,
                                1.0, label);
        }
    }
}

/** Test special case: n_outliers=0; data unchanged, return 0. */
void test_add_outliers_n_outliers_0() {
    float points_x[N_POINTS], points_y[N_POINTS];
    float params[] = {0.0f, 1.0f};
    make_inliers(points_x, points_y, N_POINTS, params, N_PARAMS, X_MIN, X_MAX);
    int ret = add_outliers(points_x, points_y, N_POINTS, 0, params, N_PARAMS, STD);
    assert_equal_int(ret, 0, "add_outliers n_outliers=0 returns 0");
}

/** Test special case: y_min==y_max (slope=0); inliers unchanged, outliers outside band, return 0. */
void test_add_outliers_y_min_eq_y_max() {
    float points_x[N_POINTS], points_y[N_POINTS], copoints_y_y[N_INLIERS];
    float params[] = {0.0f, 0.0f};
    make_inliers(points_x, points_y, N_INLIERS, params, N_PARAMS, X_MIN, X_MAX);
    memcpy(copoints_y_y, points_y, N_INLIERS * sizeof(float));
    int ret = add_outliers(points_x, points_y, N_INLIERS, N_OUTLIERS, params, N_PARAMS, STD);
    assert_equal_int(ret, 0, "add_outliers y_min==y_max returns 0");
    char label[128];
    for (int i = 0; i < N_POINTS; i++) {
        if (i < N_INLIERS) {
            snprintf(label, sizeof(label), "add_outliers y_min==y_max: inlier points_y[%d] changed", i);
            assert_almost_equal(points_y[i], copoints_y_y[i], label);
        } else {
            snprintf(label, sizeof(label), "add_outliers y_min==y_max: outlier points_y[%d] inside band", i);
            assert_almost_equal(fabs(points_y[i]) > 2*STD ? 1.0 : 0.0, 1.0, label);
        }
    }
}

/** Test edge case: n_inliers < 0, should return -1. */
void test_add_outliers_n_inliers_lt_0() {
    float points_x[N_INLIERS], points_y[N_INLIERS];
    float params[] = {0.0f, 1.0f};
    int ret = add_outliers(points_x, points_y, -1, N_OUTLIERS, params, N_PARAMS, STD);
    assert_equal_int(ret, -1, "add_outliers n_inliers<0 returns -1");
}

/** Test edge case: n_outliers < 0, should return -1. */
void test_add_outliers_n_outliers_lt_0() {
    float points_x[N_INLIERS], points_y[N_INLIERS];
    float params[] = {0.0f, 1.0f};
    int ret = add_outliers(points_x, points_y, N_INLIERS, -1, params, N_PARAMS, STD);
    assert_equal_int(ret, -1, "add_outliers n_outliers<0 returns -1");
}

/** Test edge case: n_inliers + n_outliers < 2, should return -1. */
void test_add_outliers_n_points_lt_2() {
    float points_x[N_INLIERS], points_y[N_INLIERS];
    float params[] = {0.0f, 1.0f};
    int ret = add_outliers(points_x, points_y, 0, 1, params, N_PARAMS, STD);
    assert_equal_int(ret, -1, "add_outliers n_points<2 returns -1");
}

/** Test edge case: noise_std == 0, should return -1. */
void test_add_outliers_std_0() {
    float points_x[N_INLIERS], points_y[N_INLIERS];
    float params[] = {0.0f, 1.0f};
    int ret = add_outliers(points_x, points_y, N_INLIERS, N_OUTLIERS, params, N_PARAMS, 0.0f);
    assert_equal_int(ret, -1, "add_outliers std=0 returns -1");
}

/** Test edge case: noise_std < 0, should return -1. */
void test_add_outliers_std_lt_0() {
    float points_x[N_INLIERS], points_y[N_INLIERS];
    float params[] = {0.0f, 1.0f};
    int ret = add_outliers(points_x, points_y, N_INLIERS, N_OUTLIERS, params, N_PARAMS, -1.0f);
    assert_equal_int(ret, -1, "add_outliers std<0 returns -1");
}

/** Test edge case: n_params < 2, should return -1. */
void test_add_outliers_n_params_lt_2() {
    float points_x[N_INLIERS], points_y[N_INLIERS];
    float params[] = {0.0f, 1.0f};
    int ret = add_outliers(points_x, points_y, N_INLIERS, N_OUTLIERS, params, 1, STD);
    assert_equal_int(ret, -1, "add_outliers n_params<2 returns -1");
}

/** Test edge case: x_min == x_max (all points_x equal), should return -1. */
void test_add_outliers_x_min_eq_x_max() {
    float points_x[N_INLIERS], points_y[N_INLIERS];
    float params[] = {0.0f, 1.0f};
    make_inliers(points_x, points_y, N_INLIERS, params, N_PARAMS, X_MIN, X_MAX);
    for (int i = 0; i < N_INLIERS; i++) points_x[i] = X_MAX;
    int ret = add_outliers(points_x, points_y, N_INLIERS, N_OUTLIERS, params, N_PARAMS, STD);
    assert_equal_int(ret, -1, "add_outliers x_min==x_max returns -1");
}


/* ================================================================
 * ADD_STRUCTURAL_BIAS
 * ================================================================
 * Happoints_y paths:
 *   bias_fn = 0           points_y unchanged
 *   bias_fn = 1.0         points_y[i] increased by 1.0
 *   bias_fn = 0.5*x       points_y[i] increased by 0.5*x
 *   bias_fn = sin(x)      points_y[i] increased by sin(x)
 * Edge cases:
 *   bias_fn is NULL   should return -1
 *   pr < 0            should return -1
 *   pr > 1            should return -1
 *   n_inliers < 2     should return -1
 */

static float _bias_const_0(float x)  { return 0.0f; }
static float _bias_const_1(float x)  { return 1.0f; }
static float _bias_linear(float x)   { return 0.5f * x; }
static float _bias_periodic(float x) { return sinf(x); }

/** Helper: make N_POINTS inliers on y=x, copoints_y points_y, apply bias, return copies. */
static void setup_bias_test(float *points_x, float *points_y, float *copoints_y_y) {
    float params[] = {0.0f, 1.0f};
    make_inliers(points_x, points_y, N_POINTS, params, N_PARAMS, X_MIN, X_MAX);
    memcpy(copoints_y_y, points_y, N_POINTS * sizeof(float));
}

/** Test happoints_y path: bias=0; points_y unchanged. */
void test_add_structural_bias_const_0() {
    float points_x[N_POINTS], points_y[N_POINTS], copoints_y_y[N_POINTS];
    setup_bias_test(points_x, points_y, copoints_y_y);
    int ret = add_structural_bias(points_y, points_x, N_POINTS, PR, _bias_const_0);
    assert_equal_int(ret, 0, "add_structural_bias bias=0 returns 0");
    char label[128];
    for (int i = 0; i < N_POINTS; i++) {
        snprintf(label, sizeof(label), "add_structural_bias bias=0: points_y[%d] changed", i);
        assert_almost_equal(points_y[i], copoints_y_y[i], label);
    }
}

/** Test happoints_y path: bias=1; all points_y[i] increased by 1. */
void test_add_structural_bias_const_1() {
    float points_x[N_POINTS], points_y[N_POINTS], copoints_y_y[N_POINTS];
    setup_bias_test(points_x, points_y, copoints_y_y);
    int ret = add_structural_bias(points_y, points_x, N_POINTS, PR, _bias_const_1);
    assert_equal_int(ret, 0, "add_structural_bias bias=1 returns 0");
    char label[128];
    for (int i = 0; i < N_POINTS; i++) {
        snprintf(label, sizeof(label), "add_structural_bias bias=1: points_y[%d] wrong", i);
        assert_almost_equal(points_y[i], copoints_y_y[i] + 1.0f, label);
    }
}

/** Test happoints_y path: bias=0.5*x; all points_y[i] increased by 0.5*points_x[i]. */
void test_add_structural_bias_linear() {
    float points_x[N_POINTS], points_y[N_POINTS], copoints_y_y[N_POINTS];
    setup_bias_test(points_x, points_y, copoints_y_y);
    int ret = add_structural_bias(points_y, points_x, N_POINTS, PR, _bias_linear);
    assert_equal_int(ret, 0, "add_structural_bias bias=linear returns 0");
    char label[128];
    for (int i = 0; i < N_POINTS; i++) {
        snprintf(label, sizeof(label), "add_structural_bias bias=linear: points_y[%d] wrong", i);
        assert_almost_equal(points_y[i], copoints_y_y[i] + 0.5f * points_x[i], label);
    }
}

/** Test happoints_y path: bias=sin(x); all points_y[i] increased by sin(points_x[i]). */
void test_add_structural_bias_periodic() {
    float points_x[N_POINTS], points_y[N_POINTS], copoints_y_y[N_POINTS];
    setup_bias_test(points_x, points_y, copoints_y_y);
    int ret = add_structural_bias(points_y, points_x, N_POINTS, PR, _bias_periodic);
    assert_equal_int(ret, 0, "add_structural_bias bias=sin returns 0");
    char label[128];
    for (int i = 0; i < N_POINTS; i++) {
        snprintf(label, sizeof(label), "add_structural_bias bias=sin: points_y[%d] wrong", i);
        assert_almost_equal(points_y[i], copoints_y_y[i] + sinf(points_x[i]), label);
    }
}

/** Test edge case: bias_fn == NULL, should return -1. */
void test_add_structural_bias_null() {
    float points_x[N_POINTS], points_y[N_POINTS], copoints_y_y[N_POINTS];
    setup_bias_test(points_x, points_y, copoints_y_y);
    int ret = add_structural_bias(points_y, points_x, N_POINTS, PR, NULL);
    assert_equal_int(ret, -1, "add_structural_bias NULL returns -1");
}

/** Test edge case: pr < 0, should return -1. */
void test_add_structural_bias_pr_lt_0() {
    float points_x[N_POINTS], points_y[N_POINTS], copoints_y_y[N_POINTS];
    setup_bias_test(points_x, points_y, copoints_y_y);
    int ret = add_structural_bias(points_y, points_x, N_POINTS, -1, _bias_const_1);
    assert_equal_int(ret, -1, "add_structural_bias pr<0 returns -1");
}

/** Test edge case: pr > 1, should return -1. */
void test_add_structural_bias_pr_gt_1() {
    float points_x[N_POINTS], points_y[N_POINTS], copoints_y_y[N_POINTS];
    setup_bias_test(points_x, points_y, copoints_y_y);
    int ret = add_structural_bias(points_y, points_x, N_POINTS, 1.1f, _bias_const_1);
    assert_equal_int(ret, -1, "add_structural_bias pr>1 returns -1");
}

/** Test edge case: n_inliers < 2, should return -1. */
void test_add_structural_bias_n_inliers_1() {
    float points_x[N_POINTS], points_y[N_POINTS], copoints_y_y[N_POINTS];
    setup_bias_test(points_x, points_y, copoints_y_y);
    int ret = add_structural_bias(points_y, points_x, 1, PR, _bias_const_1);
    assert_equal_int(ret, -1, "add_structural_bias n<2 returns -1");
}


/* ================================================================
 * MAIN
 * ================================================================ */

int main() {
    srand(time(NULL));

    printf("***** RUNNING TESTS FOR MAKE_INLIERS *****\n");
    test_make_inliers_slope_1_intercept_0();
    test_make_inliers_slope_neg1_intercept_n();
    test_make_inliers_slope_float_intercept_0();
    test_make_inliers_slope_1_intercept_neg_float();
    test_make_inliers_slope_0_intercept_0();
    test_make_inliers_quadratic_model();
    test_make_inliers_edge_n_lt_2();
    test_make_inliers_edge_xmin_eq_xmax();
    test_make_inliers_edge_n_params_lt_2();

    printf("***** RUNNING TESTS FOR ADD_GAUSSIAN_NOISE *****\n");
    test_add_gaussian_noise_std_int();
    test_add_gaussian_noise_std_float();
    test_add_gaussian_noise_n_inliers_lt_2();
    test_add_gaussian_noise_std_0();
    test_add_gaussian_noise_std_lt_0();

    printf("***** RUNNING TESTS FOR ADD_OUTLIERS *****\n");
    test_add_outliers_happoints_y_path();
    test_add_outliers_n_outliers_0();
    test_add_outliers_n_inliers_lt_0();
    test_add_outliers_n_outliers_lt_0();
    test_add_outliers_n_points_lt_2();
    test_add_outliers_std_0();
    test_add_outliers_std_lt_0();
    test_add_outliers_n_params_lt_2();
    test_add_outliers_x_min_eq_x_max();
    test_add_outliers_y_min_eq_y_max();

    printf("***** RUNNING TESTS FOR ADD_STRUCTURAL_BIAS *****\n");
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