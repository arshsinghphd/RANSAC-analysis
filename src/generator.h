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

/**
 * Adds zero mean gaussian noise to points_y in place.
 * 
 * Params:
 * points_y    a list of floats, modified in place
 * n_inliers   int, number of points to modify
 * std         float, standard deviation of gaussian noise
 * 
 * Returns:
 * 0 for success
 * -1 for error if n_inliers < 2
 * -1 for error if std <= 0
 */
int add_gaussian_noise(float* points_y, int n_inliers, float std);

/**
 * Appends n_outliers points to points_x and points_y in place, guaranteeing 
 * that every added point lies outside the inlier band defined by the true 
 * polynomial model and noise_std. Outliers are placed in the x-range of inliers
 * and in a y range that is guaranteed to be outside the inlier band defined as 
 * (2 * noise_std) from the model value at that x.
 * 
 * Params:
 * points_x     a list of floats of size n_inliers, modified in place
 * points_y     a list of floats of size n_inliers, modified in place
 * n_inliers    int, number of existing valid points
 * n_outliers   int, number of outliers to append
 * params       a list of n_params floats, polynomial coefficients
 *              from lowest to highest degree [a0, a1, ...]
 * n_params     int, number of model parameters
 * noise_std    float, standard deviation of inlier noise, used to
 *              define the inlier band as 2 * noise_std from the model
 *
 * Returns:
 * 0 for success
 * -1 for error if n_inliers < 0
 * -1 for error if n_outliers < 0
 * -1 for error if n_inliers + n_outliers < 2
 * -1 for error if noise_std <= 0
 * -1 for error if n_params < 2
 */
int add_outliers(float* points_x, float* points_y, int n_inliers, 
    int n_outliers, float* params, int n_params, float noise_std);

/**
 * Adds structural bias to the data inplace based on user's inputs.
 * 
 * Params:
 * points_y    a list of floats
 * points_x    a list of floats
 * n_inliers   int, number of inliers to be modified
 * bias_fn     a function or lambda that defines structural bias
 * e.g. 
 *      lambda x: 0.5 * x       # linear structural bias
 *      lambda x: math.sin(x)   # periodic structural bias
 * 
 * Returns:
 * 0 for success
 * -1 for error
 */
 int add_structural_bias(float* points_y, float* points_x, int n_inliers, 
    float pr, float (*bias_fn)(float));

 #endif