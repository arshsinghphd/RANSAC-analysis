/** Implementation of generator.h */
#include "generator.h"
#include<math.h>
#include<stdio.h>
#include<stdlib.h>

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
                float x_min, float x_max) {
    if(n_inliers < 2 || x_min == x_max || n_params < 2){
            return -1;
        }
    float step = (float) (x_max - x_min) / (n_inliers - 1);
    float x;
    float y;
    for(int i = 0; i < n_inliers; i++) {
        x = x_min + i * step;
        y = 0.0;
        for(int j = 0; j < n_params; j++){
            y += params[j] * pow(x, j);
        }
        points_x[i] = x;
        points_y[i] = y;
    }
    return 0;
}

/**
 * Helper function for add_gaussian_noise. Returns a random number drawn from a zero mean gaussian distribution
 * with standard deviation sigma, using the Box-Muller transform.
 * This function is tested indirected via add_gaussian_noise.
 * 
 * Source:
 * Box, G. E. P. and Muller, M. E. 1958. A note on the generation of random 
 * normal deviates. The Annals of Mathematical Statistics 29, 2, 610-611.
 * 
 * Params:
 * sigma    float, standard deviation
 * 
 * Returns:
 * float,   random sample from N(0, sigma^2)
 */
static float _box_muller(float std) {
    float u1 = (float) rand() / RAND_MAX;
    float u2 = (float) rand() / RAND_MAX;
    float z = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
    return z * std;
}

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
int add_gaussian_noise(float* points_y, int n_inliers, float std) {
    if (n_inliers < 2 || std <= 0)
        return -1;
    for(int i = 0; i < n_inliers; i++)
        points_y[i] += _box_muller(std);
    return 0;
}

