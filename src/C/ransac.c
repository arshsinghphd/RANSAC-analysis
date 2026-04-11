/* Implementation of RANSAC.H */

#include "ransac.h"
#include "model.h"

#include<math.h>
#include<stdio.h>
#include<stdlib.h>


/* =============================================================================
Helper: Computes the population mean of distances. 
Tested indirectly via estimate_epsilon.

Params:
    distances   a list of n_points floats
    n_points    int, number of distances to process

Returns:
    float, mean of distances
============================================================================= */
float _estimate_mean(float* distances, int n_points) {
	float mean = 0.0;
	for (int i = 0; i < n_points; i++) {
		mean += distances[i];
	}
	return (float) mean / n_points;
}


/* =============================================================================
Helper: Computes the population standard deviation of distances given a 
precomputed mean. Uses population standard deviation dividing by n_points rather
than n_points - 1, which is appropriate for threshold estimation as recommended
by Fischler and Bolles (1981). Tested indirectly via estimate_epsilon.

Params:
    distances   a list of n_points floats
    n_points    int, number of distances to process
    mean        float, precomputed mean of distances

Returns:
    float, population standard deviation of distances
============================================================================= */
float _estimate_std(float* distances, int n_points, float mean) {
	float variance = 0.0;
    for (int i = 0; i < n_points; i++) {
        variance += (distances[i] - mean) * (distances[i] - mean);
    }
    return (float) sqrt(variance / n_points);
}


/* =============================================================================
Helper: Fills the array of residuals.

Params:
	points_x    	a pointer to a list of n_points floats
    points_y    	a pointer to a list of n_points floats
    n_points    	int, number of points
	n_params		int, min parameters to be estimated in the model
    params      	a pointer to a list of floats containing model coefficients
    mean        	float, precomputed mean of distances
    residuals 		a pointer to list of residuals to be filled in place

Returns:
	nothing 
============================================================================= */
void _fill_residuals(float* points_x, float* points_y, int n_points, 
	float* params, int n_params, float* residuals) {
	float y_model = 0;
    for (int i = 0; i < n_points; i++) {
    	y_model = eval_model(points_x[i], params, n_params);
    	residuals[i] = fabs(points_y[i] - y_model);
    }
}

/* =============================================================================
Estimates the inlier threshold t from the residual distribution of a preliminary
least squares fit computed from the data itself. Fits a polynomial to all points
using least squares, computes the vertical residual of each point from that 
model, then returns mean + 2 * std of all residuals, consistent with the 
recommendation of Fischler and Bolles (1981).

Like estimate_epsilon, this estimate is a rough first guess only. At high
outlier fractions the preliminary least squares fit is corrupted, inflating
mean and std and producing an overly large threshold that may accept outliers
as inliers. It should be treated as a starting point only.

Params:
    points_x    a list of n_points floats
    points_y    a list of n_points floats
    n_points    int, number of points
    n_params	int, min parameters to be estimated in the model

Returns:
    float, estimated inlier threshold t
    -1 for error if n_points < 2
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

/* =============================================================================
Estimates the outlier fraction epsilon from the residual distribution of a
preliminary least squares fit computed from the data itself. Fits a line to all
points using least squares, computes the vertical residual of each point from
that line, then returns the fraction of outliers as fraction of points whose 
residual exceeds mean + 2 * std of all residuals.

This estimate is a rough first guess only. It is unreliable at high outlier
fractions because outliers corrupt the preliminary least squares fit, inflating
mean and std and causing the function to undercount outliers. This is a known
limitation — the same problem RANSAC is designed to solve. 

A more robust approach is iterative refinement: start with a conservative 
epsilon such as 0.5, run RANSAC, observe the inlier fraction of the best model,
update epsilon, and repeat until convergence. This is left as future work.

In this project epsilon is known exactly from the synthetic data generation
process, so this function is provided for completeness and for use in real-world
scenarios where the true epsilon is unknown.

Params:
    points_x    a list of n_points floats
    points_y    a list of n_points floats
    n_points    int, number of points
	n_params	int, min parameters to be estimated in the model
Returns:
    float, estimated outlier fraction in [0, 1], rough estimate only
    -1 for error if n_points < 2
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
Computes the number of RANSAC iterations k required to guarantee that at
least one clean sample is drawn with probability 1 - failure_prob using the 
analytical formula:

    k = ceil(log(failure_prob) / log(1 - (1 - epsilon)^n_params))

Params:
    epsilon         float, estimated outlier fraction in [0, 1)
    n_params        int, minimum number of points to fit the model
    failure_prob    float, acceptable failure probability, default 0.01

Returns:
    int, 	number of iterations k rounded up (ceiling) to nearest integer
     1 		if epsilon == 0, clean sample requires only one iteration
    -1 		for error if epsilon < 0 or epsilon >= 1
    -1 		for error if n_params < 2
    -1 		for error if failure_prob <= 0 or failure_prob >= 1
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
Computes the expected inlier count d as floor((1 - epsilon) * n_points).
    This is consistent with the same epsilon used to compute k, ensuring both
    parameters reflect a coherent assumption about the data.

    Params:
        epsilon     float, estimated outlier fraction in (0, 1)
        n_points    int, total number of points

    Returns:
        int, expected inlier count d
        -1 for error if epsilon <= 0 or epsilon >= 1
        -1 for error if n_points < 2
============================================================================= */
int compute_d(float epsilon, int n_points) {
	if (epsilon <= 0 || epsilon >= 1 || n_points < 2)
		return -1;
	return floorf((1 - epsilon) * (float) n_points);
}

