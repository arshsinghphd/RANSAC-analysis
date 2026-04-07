#ifndef GENERATOR_H
#define GENERATOR_H

/**
 * Functions for generating testing data for RANSAC pipeline. 
 * Supports polynomial models of any degree via params and n_params. 
 * Provides inlier generation, adding noise, outlier generation, and 
 * adding structural bias.
 * 
 * Model parameters are stored as polynomial coefficients from lowest to highest
 * degree:
 * params[0] = a0  (intercept / constant term)
 * params[1] = a1  (slope / linear term)
 * params[2] = a2  (quadratic term)
 * ...
 * 
 * For line fitting n_params = 2, giving y = a0 + a1 * x.
 * For quadratic fitting n_params = 3, giving y = a0 + a1 * x + a2 * x^2.
 * 
 * Functions:
 * make_inliers            fills points_x, points_y with inlier data
 * box_muller              generates gaussian random numbers
 * add_gaussian_noise      adds zero mean gaussian noise to points_y
 * add_outliers            appends outlier points to points_x, points_y
 * add_structural_bias     adds structural bias to points_y
 */

/* function declarations */

/**
 * Fills points_x and points_y in place with n_inliers points sampled uniformly 
 * from x_min to x_max, based on the polynomial model defined by params and 
 * n_params:
 * y = a0 + a1 * x + a2 * x^2 + ... + a_{n-1} * x^{n-1}
 * 
 * Params:
 * points_x    a list of floats of size >= n_inliers, modified in place
 * points_y    a list of floats of size >= n_inliers, modified in place
 * n_inliers   int, number of inlier points to generate
 * params      a list of n_params floats, polynomial coefficients
 *              from lowest to highest degree [a0, a1, ...]
 * n_params    int, number of model parameters
 * x_min       float, lower limit of x
 * x_max       float, upper limit of x
 * 
 * Returns:
 * 0 for success
 * -1 for error if n_inliers < 2
 * -1 for error if x_min == x_max
 * -1 for error if n_params < 2
 */
int make_inliers(float* points_x, float* points_y, int n_inliers, 
                float* params, int n_params,
                float x_min, float x_max);



#endif