/** Implementation of generator.h */
#include "generator.h"
#include<math.h>
#include<stdio.h>
#include<stdlib.h>

/**
 * make_inliers
 * 
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
 * _box_muller
 * 
 * Helper function for add_gaussian_noise. Returns a random number drawn from a
 * zero mean gaussian distribution with standard deviation sigma, using the 
 * Box-Muller transform. This function is tested indirected via 
 * add_gaussian_noise.
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
    float u1 = (float) rand() / (float) RAND_MAX;
    float u2 = (float) rand() / (float) RAND_MAX;
    float z = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
    return z * std;
}

/**
 * add_gaussain_noise
 * 
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

/**
 * add_outliers
 * 
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
int add_outliers(float* points_x, float* points_y, 
    int n_inliers, int n_outliers, float* params, int n_params, 
    float noise_std) 
{
    if (n_inliers < 0 || n_outliers < 0 ||
        n_inliers + n_outliers < 2 ||
        noise_std <= 0.0f || n_params < 2)
        return -1;
    // find x_min, x_max, y_min, y_max
    float x_min = points_x[0];
    float x_max = points_x[0];
    float y_min = points_y[0];
    float y_max = points_y[0];
    for (int i = 0; i < n_inliers; i++) {
        if(points_x[i] < x_min)
            x_min = points_x[i];
        if(points_x[i] > x_max) 
            x_max = points_x[i];
        if(points_y[i] < y_min)
            y_min = points_y[i];
        if(points_y[i] > y_max) 
            y_max = points_y[i];
    }
    // estimate x_range
    float x_range;
    if (x_max - x_min > 1e-6)
        x_range = x_max - x_min;    
    else 
        return -1;  // if x_min == x_max, no outliers can be added
    // variables to estimate outlier y
    float inlier_band = 2 * noise_std;
    float y_range;
    // estimate y_range
    if (y_max - y_min > 1)
        y_range = y_max - y_min;  
    else 
        y_range = 1;  // means no noise in data, must add outliers though
    // add outliers to points_x, points_y
    float x, y, y_on_model;
    int idx = n_inliers;
    for (int i = 0; i < n_outliers; i++) {
        // pick random x on x_range
        x = x_min + ((float) rand() / (float) RAND_MAX) * x_range;
        // estimate corresponding y on model
        y_on_model = 0.0;
        for (int j = 0; j < n_params; j++) {
            y_on_model += params[j] * pow(x, j);
        }
        // 50% change above model. 50% chance below model
        // outliers are some random fraction of the range of y outside the band
        if (rand()/RAND_MAX < 0.5)
            y = y_on_model + inlier_band + ((float) rand()/ (float) RAND_MAX) * y_range;
        else
            y = y_on_model - inlier_band - ((float) rand()/ (float) RAND_MAX) * y_range;
        points_x[idx] = x;
        points_y[idx] = y;
        idx++;
    }
    return 0;
}

/**
 * add_structural_bias
 * 
 * Adds structural bias to the data inplace based on user's inputs with a given 
 * probability.
 * 
 * Params:
 * points_y     a list of floats
 * points_x     a list of floats
 * n_inliers    int, number of inliers to be modified
 * pr           float, probability with which the bias occurs
 * bias_fn      a function or lambda that defines structural bias
 * e.g. 
 *      lambda x: 0.5 * x       # linear structural bias
 *      lambda x: math.sin(x)   # periodic structural bias
 * 
 * Returns:
 * 0 for success
 * -1 for error
 */
 int add_structural_bias(float* points_y, float* points_x, int n_inliers, 
    float pr, float (*bias_fn)(float)) {
    if (n_inliers < 2 || bias_fn == NULL || pr < 0 || pr > 1) {
        return -1;
    }
    for (int i = 0; i < n_inliers; i++) {
        if (pr > (float) rand() / (float) RAND_MAX) {
            points_y[i] += bias_fn(points_x[i]);
        }
    }
    return 0;
 }
