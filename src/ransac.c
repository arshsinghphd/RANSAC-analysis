/* Implementation of RANSAC.H */

#include "ransac.h"
#include "model.h"

#include<math.h>
#include<stdio.h>
#include<stdlib.h>


/* =============================================================================
 * HELPERS 
============================================================================= */

/** ----------------------------------------------------------------------------
 * Helper: computes the population mean of distances.
 * Tested indirectly via estimate_epsilon.
 *
 * Params:
 *     distances   array of n_points floats
 *     n_points    int, number of distances to process
 *
 * Returns:
 *     float, mean of distances
----------------------------------------------------------------------------- */

float _estimate_mean(float* distances, int n_points) {
	float mean = 0.0;
	for (int i = 0; i < n_points; i++) {
		mean += distances[i];
	}
	return (float) mean / n_points;
}


/** ----------------------------------------------------------------------------
 * Helper: computes the population standard deviation of distances given
 * a precomputed mean. Uses population standard deviation dividing by
 * n_points rather than n_points - 1, which is appropriate for threshold
 * estimation as recommended by Fischler and Bolles (1981).
 * Tested indirectly via estimate_epsilon.
 *
 * Params:
 *     distances   array of n_points floats
 *     n_points    int, number of distances to process
 *     mean        float, precomputed mean of distances
 *
 * Returns:
 *     float, population standard deviation of distances
 * -------------------------------------------------------------------------- */
float _estimate_std(float* distances, int n_points, float mean) {
	float variance = 0.0;
    for (int i = 0; i < n_points; i++) {
        variance += (distances[i] - mean) * (distances[i] - mean);
    }
    return (float) sqrt(variance / n_points);
}

/** ----------------------------------------------------------------------------
 * Helper: fills the residuals array with the absolute vertical residual
 * of each point from the polynomial model defined by params.
 * Tested indirectly via estimate_epsilon and compute_t.
 *
 * Params:
 *     points_x    array of n_points floats
 *     points_y    array of n_points floats
 *     n_points    int, number of points
 *     n_params    int, number of model parameters
 *     params      array of n_params floats, model coefficients
 *     residuals   array of n_points floats, filled in place
 *
 * Returns:
 *     void
 * -------------------------------------------------------------------------- */

void _fill_residuals(float* points_x, float* points_y, int n_points, 
	float* params, int n_params, float* residuals) {
	float y_model = 0;
    for (int i = 0; i < n_points; i++) {
    	y_model = eval_model(points_x[i], params, n_params);
    	residuals[i] = fabs(points_y[i] - y_model);
    }
}

/* =============================================================================
 * compute_t
 * 
 * Estimates the inlier threshold t from the residual distribution of a
 * preliminary least squares fit computed from the data itself. Fits a
 * polynomial to all points using least squares, computes the vertical
 * residual of each point from that model, then returns mean + 2 * std
 * of all residuals, consistent with the recommendation of Fischler and
 * Bolles (1981).
 *
 * Like estimate_epsilon, this estimate is a rough first guess only. At
 * high outlier fractions the preliminary least squares fit is corrupted,
 * inflating mean and std and producing an overly large threshold that
 * may accept outliers as inliers. It should be treated as a starting
 * point only.
 *
 * Params:
 *     points_x    array of n_points floats
 *     points_y    array of n_points floats
 *     n_points    int, number of points
 *     n_params    int, number of model parameters
 *
 * Returns:
 *     float, estimated inlier threshold t
 *     -1 for error if n_points < 2
============================================================================= */
float compute_t(float* points_x, float* points_y, int n_points, int n_params) {
	if (n_points < 2) {
		return -1;
	}
	float params[n_params], residuals[n_points];
	int ret = fit_model(points_x, points_y, n_points, params, n_params);
	if (ret == -1)
		return -1;
	_fill_residuals(points_x, points_y, n_points, params, n_params, residuals);
	float mean = _estimate_mean(residuals, n_points);
    float std = _estimate_std(residuals, n_points, mean);
    return (float) mean + 2 * std;
}

/** ============================================================================
 * estimate_epsilon
 *
 * Estimates the outlier fraction epsilon from the residual distribution
 * of a preliminary least squares fit computed from the data itself.
 * Fits a polynomial to all points using least squares, computes the
 * vertical residual of each point from that model, then returns the
 * fraction of points whose residual exceeds mean + 2 * std of all
 * residuals.
 *
 * This estimate is a rough first guess only. It is unreliable at high
 * outlier fractions because outliers corrupt the preliminary least
 * squares fit, inflating mean and std and causing the function to
 * undercount outliers. This is a known limitation — the same problem
 * RANSAC is designed to solve. A more robust approach is iterative
 * refinement: start with a conservative epsilon such as 0.5, run
 * RANSAC, observe the inlier fraction of the best model, update
 * epsilon, and repeat until convergence. This is left as future work.
 *
 * In this project epsilon is known exactly from the synthetic data
 * generation process, so this function is provided for completeness
 * and for use in real-world scenarios where the true epsilon is
 * unknown.
 *
 * Params:
 *     points_x    array of n_points floats
 *     points_y    array of n_points floats
 *     n_points    int, number of points
 *     n_params    int, number of model parameters
 *
 * Returns:
 *     float, estimated outlier fraction in [0, 1], rough estimate only
 *     -1 for error if n_points < 2
============================================================================= */
float estimate_epsilon(float* points_x, float* points_y, int n_points, 
	int n_params) 
{
	if (n_points < n_params){
        return -1;
	}
    float params[n_params], residuals[n_points];
    int ret = fit_model(points_x, points_y, n_points, params, n_params);
    if(ret == -1)
        return -1;
    _fill_residuals(points_x, points_y, n_points, params, n_params, residuals);
    float mean = _estimate_mean(residuals, n_points);
    float std = _estimate_std(residuals, n_points, mean);
    float threshold = mean + 2 * std;

    int count_outliers = 0;
    for (int i = 0; i < n_points; i++) {
    	// definition of outlier
    	if (residuals[i] > threshold){
    		count_outliers++;
    	}
    }
    return (float) count_outliers / n_points;
}


/* =============================================================================
 * compute_k
 *
 * Computes the number of RANSAC iterations k required to guarantee that
 * at least one clean sample is drawn with probability 1 - failure_prob
 * using the analytical formula:
 *
 *     k = ceil(log(failure_prob) / log(1 - (1 - epsilon)^n_params))
 *
 * Params:
 *     epsilon         float, estimated outlier fraction in [0, 1)
 *     n_params        int, minimum number of points to fit the model
 *     failure_prob    float, acceptable failure probability
 *
 * Returns:
 *     int, number of iterations k rounded up to nearest integer
 *     1   if epsilon == 0, clean sample requires only one iteration
 *     -1  for error if epsilon < 0 or epsilon >= 1
 *     -1  for error if n_params < 2
 *     -1  for error if failure_prob <= 0 or failure_prob >= 1
============================================================================= */
int compute_k(float epsilon, int n_params, float failure_prob) {
	// if no outliers 1 iteration is all we need
	if (epsilon == 0)
		return 1;
	if (epsilon < 0 || epsilon >= 1 || n_params < 2 || 
		failure_prob <= 0 || failure_prob >= 1)
		return -1;
	return (int) ceilf(	logf(failure_prob) / 
						logf(1.0f - powf(1.0f - epsilon, n_params))
						);
}

/* =============================================================================
 * compute_d
 *
 * Computes the expected inlier count d as floor((1 - epsilon) * n_points).
 * This is consistent with the same epsilon used to compute k, ensuring
 * both parameters reflect a coherent assumption about the data.
 *
 * Params:
 *     epsilon     float, estimated outlier fraction in (0, 1)
 *     n_points    int, total number of points
 *
 * Returns:
 *     int, expected inlier count d
 *     -1  for error if epsilon <= 0 or epsilon >= 1
 *     -1  for error if n_points < 2
============================================================================= */
int compute_d(float epsilon, int n_points) {
	if (epsilon <= 0 || epsilon >= 1 || n_points < 2)
		return -1;
	return floorf((1 - epsilon) * (float) n_points);
}


/* =============================================================================
 * fisher_yates
 *
 * Partial Fisher-Yates in-place shuffle. Draws m random indices from
 * [0, n) into idx[0..m-1]. Only the first m positions are shuffled
 * at O(m) cost.
 *
 * Params:
 *     idx     pointer to int array of size n, initialized to [0..n-1]
 *     n       int, total number of elements
 *     m       int, number of indices to draw
 *
 * Returns:
 *     void, modifies idx in place
 *
 * Reference:
 *     Durstenfeld, R. 1964. Algorithm 235: Random permutation.
 *     Communications of the ACM 7, 7, 420.
============================================================================= */
void fisher_yates(int* idx, int n, int m) {
    /* initialise index array 0..n-1 */
    for (int j = 0; j < n; j++)
        idx[j] = j;
    /* partial shuffle — only first m positions needed */
    for (int j = 0; j < m; j++) {
    	// random in [j, n)
        int k   = j + rand() % (n - j);
        // swap idx[j] and idx[k]
        int tmp = idx[j];                       
        idx[j]  = idx[k];
        idx[k]  = tmp;
    }
}


/* =============================================================================
 * ransac
 *
 * Finds the best fitting polynomial model from noisy data containing
 * outliers using the Random Sample Consensus algorithm. Randomly samples
 * n_params points k_resample times, fits a candidate model to each
 * sample, counts inliers within threshold vertical residual, and tracks
 * the best model found. Terminates early if a model with at least
 * expected_inliers inliers is found. Refits the final model on all
 * inliers of the best consensus set.
 *
 * return_array layout (modified in place):
 *     return_array[0]               number of inliers in best model
 *     return_array[1]               number of iterations actually run
 *     return_array[2..2+n_params-1] best model params (a0, a1, ...)
 *
 * Params:
 *     points_x            array of n_points floats
 *     points_y            array of n_points floats
 *     n_points            int, total number of points
 *     n_params            int, minimum number of points to fit the model
 *     k_resample          int, maximum number of iterations
 *     threshold           float, inlier vertical residual threshold t
 *     expected_inliers    int, early stop threshold d
 *     return_array        array of at least 2 + n_params floats,
 *                         modified in place
 *
 * Returns:
 *     0  for success
 *     -1 for error if n_points < 2
 *     -1 for error if n_params < 2
 *     -1 for error if k_resample < 1
 *     -1 for error if threshold <= 0
 *     -1 for error if expected_inliers > n_points
 *     -1 for error if n_points < n_params
============================================================================= */


/* -----------------------------------------------------------------------------
 * Helper: runs one RANSAC iteration. Samples n_params points, fits a
 * model, counts inliers via vertical residual. Updates best_params and
 * best_inliers if this iteration produced a better model.
 *
 * Tested indirectly via ransac() with k = 1.
 *
 * Params:
 *     points_x        array of n_points floats
 *     points_y        array of n_points floats
 *     n_points        int, total number of points
 *     n_params        int, number of model parameters
 *     threshold       float, inlier vertical residual threshold
 *     best_params     array of n_params floats, updated in place
 *     best_inliers    int pointer, updated in place
 *
 * Returns:
 *     1   if best model was updated
 *     0   if no improvement
 *     -1  if fit failed on degenerate sample
----------------------------------------------------------------------------- */
static int _ransac_iteration(float* points_x, float* points_y,
                              int n_points, int n_params, float threshold,
                              float* best_params, int* best_inliers) {
    int idx[n_points];
    fisher_yates(idx, n_points, n_params);

    /* collect sampled points */
    float sample_x[n_params], sample_y[n_params];
    for (int j = 0; j < n_params; j++) {
        sample_x[j] = points_x[idx[j]];
        sample_y[j] = points_y[idx[j]];
    }

    /* fit model to sample — return -1 if degenerate */
    float params[n_params];
    int ret = fit_model(sample_x, sample_y, n_params, params, n_params);
    if (ret == -1)
        return -1;

    /* count inliers using vertical residual */
    float inliers_x[n_points], inliers_y[n_points];
    int   n_inliers = 0;
    find_model_inliers(points_x, points_y, n_points,
                       params, n_params, threshold,
                       inliers_x, inliers_y, &n_inliers);

    /* update best model if improved */
    if (n_inliers > *best_inliers) {
        *best_inliers = n_inliers;
        for (int j = 0; j < n_params; j++)
            best_params[j] = params[j];
        return 1; /* best updated */
    }
    return 0; /* no improvement */
}


/* -----------------------------------------------------------------------------
 * Helper: collects inliers of best model and refits on full consensus
 * set. Stores result in params, modifying it in place.
 *
 * Tested indirectly via ransac(). Relies on find_model_inliers and
 * fit_model, both of which have been tested thoroughly.
 *
 * Params:
 *     points_x        array of n_points floats
 *     points_y        array of n_points floats
 *     n_points        int, total number of points
 *     best_params     array of n_params floats, best model from loop
 *     n_params        int, number of model parameters
 *     threshold       float, inlier vertical residual threshold
 *     params          array of n_params floats, result stored here
 *
 * Returns:
 *     0   for success
 *     -1  if refit fails
 * -------------------------------------------------------------------------- */
static int _final_refit(float* points_x, float* points_y, int n_points,
                         float* best_params, int n_params, float threshold,
                         float* params) {
    /* collect all inliers of best model */
    float inlier_x[n_points], inlier_y[n_points];
    int   n_inliers = 0;
    find_model_inliers(points_x, points_y, n_points,
                       best_params, n_params, threshold,
                       inlier_x, inlier_y, &n_inliers);

    /* refit on all inliers */
    return fit_model(inlier_x, inlier_y, n_inliers, params, n_params);
}

/* =============================================================================
 * RANSAC
============================================================================= */
int ransac(float* points_x, float* points_y, int n_points, int n_params,
           int k_resample, float threshold, int expected_inliers,
           float* return_array) {
    /* guard conditions */
    if (n_points < 2 || n_params < 2 || n_points < n_params ||
            k_resample < 1 || threshold <= 0.0f ||
            expected_inliers > n_points){
        return -1;
	}
    int   best_inliers  = 0;
    int   iterations_run = 0;
    float best_params[n_params];
    for (int j = 0; j < n_params; j++)
        best_params[j] = 0.0f;

    for (int i = 0; i < k_resample; i++) {
        _ransac_iteration(points_x, points_y, n_points, n_params,
                          threshold, best_params, &best_inliers);
        iterations_run++;
        // early stop
        if (best_inliers >= expected_inliers)
            break;
    }

    if (best_inliers == 0)
        return -1;

    /* final refit on best consensus set */
    float params[n_params];
    int ret = _final_refit(points_x, points_y, n_points,
                           best_params, n_params, threshold, params);
    //printf("params[0]=%f, params[1]=%f\n", params[0], params[1]);

    if (ret == -1)
        return -1;

    /* fill return_array */
    return_array[0] = (float) best_inliers;
    return_array[1] = (float) iterations_run;
    for (int j = 0; j < n_params; j++)
        return_array[2 + j] = params[j];

    return 0;
}
