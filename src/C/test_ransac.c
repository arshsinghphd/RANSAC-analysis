/**
Tests for ransac.py functions:
    estimate_epsilon
    compute_t
    compute_k
    compute_d
    ransac

estimate_epsilon estimates the outlier fraction from the residual distribution
of a preliminary least squares fit using vertical residuals from eval_model.

compute_t estimates the inlier threshold as mean + 2 * std of vertical
residuals from a preliminary least squares fit, consistent with the
recommendation of Fischler and Bolles (1981).

compute_k computes the required number of RANSAC iterations from epsilon and
n_params using the analytical formula at failure probability p = 0.01.

compute_d computes the expected inlier count as floor((1 - epsilon) * n_points),
consistent with the same epsilon used to compute k.

ransac finds the best fitting polynomial model of degree n_params - 1 from
noisy data containing outliers using the Random Sample Consensus algorithm.
It randomly samples n_params points, fits a model via fit_model, counts
inliers within threshold vertical residual using find_model_inliers, and
repeats k_resample times or until expected_inliers are found. Results are
stored in return_array in place.

return_array layout (length depends on n_params):
    return_array[0]                 n_points
    return_array[1]                 n_params
    return_array[2]                 k_resample
    return_array[3]                 threshold
    return_array[4]                 expected_inliers
    return_array[5..5+n_params-1]   best model params (a0, a1, ..., a_{n-1})
                                    from lowest to highest degree
    return_array[5+n_params]        number of inliers in best model
    return_array[6+n_params]        number of iterations actually run
*/

#include "ransac.h"
#include "generator.h"
#include "model.h"

#include<stdio.h>
#include<string.h>
#include<math.h>

/* Constants */
#define N 100          
#define X_MIN 0.0f         
#define X_MAX 99.0f        
#define TRUE_SLOPE 2.0f         
#define TRUE_INTERCEPT 5.0f         
#define NOISE_STD 0.5f         
#define N_PARAMS 2         
#define EPSILON 1e-5

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

/* Helper function tests if all values are same in two float arrays. */
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

/*	Helper: fills points_x, points_y with n inliers on the true linear model. */
static void _make_line(float* points_x, float* points_y, int n) {
    float params[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    make_inliers(points_x, points_y, n, params, N_PARAMS,
                 X_MIN, X_MAX);
}


/* 	Helper: appends n_outliers = floor(n_inliers * epsilon) outlier points to 
	points_x and points_y in place. */
static void _add_outliers(float* points_x, float* points_y,
                               int n_inliers, float epsilon) {
    int n_outliers = (int)(n_inliers * epsilon); // floor via int cast
    float params[] = {TRUE_INTERCEPT, TRUE_SLOPE};
    add_outliers(points_x, points_y, n_inliers, n_outliers,
                 params, N_PARAMS, NOISE_STD);
}


/* =============================================================================
Tests for compute_t which estimates the inlier threshold t as
mean + 2 * std of the perpendicular distances from all points to
the preliminary model line.

Happy paths:
    clean data no noise, t should be near 0.0
    gaussian noise with known std, t should be near 2 * std

Edge cases:
    n_points < 2, should return -1
============================================================================= */
void test_clean_data_t() {
	float points_x[N], points_y[N];
	float true_epsilon = 0.0f;
	float params[] = {TRUE_INTERCEPT, TRUE_SLOPE};
	_make_line(points_x, points_y, N);
	float t = compute_t(points_x, points_y, N, N_PARAMS);
	assert_almost_equal(t, 0, "clean_data");
}



/* =============================================================================
Tests for estimate_epsilon.

estimate_epsilon is a rough first guess only. At low outlier fractions the 
estimate may be reasonably close to the true value. At high outlier fractions 
the preliminary least squares fit is corrupted, inflating mean and std and 
causing the function to undercount outliers. Tests reflect this limitation by 
using loose deltas at low fractions and asserting only a valid range at high
fractions.

Happy paths:
	clean data no outliers      epsilon should be exactly 0.0
 	outlier fraction 0.20       epsilon should be within 0.2 of true
 	outlier fraction 0.40       epsilon should be within 0.4 of true
 	outlier fraction 0.60       epsilon should be in valid range [0, 1)

Edge cases:
	n_points < 2                should return -1
============================================================================= */

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
void test_estimate_epsilon_n_points_less_than_2() {
    float points_x[1], points_y[1];
    /* single point, invalid input */
    _make_line(points_x, points_y, 1); 
    float epsilon = estimate_epsilon(points_x, points_y, 1, N_PARAMS);
    assert_almost_equal(epsilon, -1.0f,
        "estimate_epsilon/n_points_less_than_2");
}

/* =============================================================================
  MAIN 
============================================================================= */
int main() {
	printf("***** RUNNING TESTS FOR ESTIMATE_EPSILON *****\n");
	test_estimate_epsilon_clean_data();
	test_estimate_epsilon_low_outlier_fraction();
	test_estimate_epsilon_medium_outlier_fraction();
	test_estimate_epsilon_n_points_less_than_2();
	printf("***** RUNNING TESTS FOR COMPUTE_T *****\n");
	test_clean_data_t();

	return 0;
}


