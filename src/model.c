/* Implementation of model.h. */

#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>


float eval_model(float x, float* params, int n_params) {
    if (n_params < 2)
        return -1;
    float y = params[0];
    for(int i = 1; i < n_params; i++)
        y += params[i] * pow(x, i);
    return y;
}


/**
 * Fits a polynomial model of degree n_params - 1 to n_points data points
 * using least squares, solved via Gaussian elimination on the normal
 * equations. Stores the n_params coefficients in params in order from
 * lowest to highest degree:
 *     params[0] = a0  (intercept)
 *     params[1] = a1  (slope for line)
 *     params[2] = a2  (quadratic term)
 *     ...
 *
 * For n_params = 2 this is equivalent to ordinary least squares (OLS):
 *     y = a0 + a1 * x
 *
 * The normal equations (X^T X) a = X^T y are formed using the Vandermonde
 * matrix X and solved using Gaussian elimination with partial pivoting and
 * back substitution. Runs in O(n_points) for n_params << n_points.
 *
 * Params:
 *     points_x    array of n_points floats
 *     points_y    array of n_points floats
 *     n_points    int, number of points to fit
 *     params      array of n_params floats, modified in place
 *     n_params    int, number of model parameters,
 *                 equals polynomial degree + 1
 *
 * Returns:
 *     0  for success
 *     -1 for error if n_points < n_params
 *     -1 for error if n_params < 2
 *     -1 for error if matrix is singular (e.g. all x values equal)
 */
int fit_model(float* points_x, float* points_y, int n_points,
              float* params, int n_params) {
    if (n_points < n_params || n_params < 2) {
        return -1;
    }

    // test if x_min == x_max
    float x_min = points_x[0];
    float x_max = points_x[0];
    for (int i = 0; i < n_points; i++) {
        if (points_x[i] < x_min)
            x_min = points_x[i];
        if (points_x[i] > x_max)
            x_max = points_x[i];
    }
    if (fabs(x_min - x_max) < 1e-4) {     // x_min == x_max with floats
        return -1;
    }
    int d = n_params;

    /* build X^T X (d x d array) and X^T y (d array) */
    float XtX[d][d];
    float Xty[d];
    for (int i = 0; i < d; i++) {
        Xty[i] = 0.0f;
        for (int j = 0; j < d; j++)
            XtX[i][j] = 0.0f;
    }

    for (int i = 0; i < n_points; i++) {
        float xi = points_x[i];
        float yi = points_y[i];

        /* precompute powers of xi: xpow[k] = xi^k */
        float xpow[2 * d];
        xpow[0] = 1.0f;
        for (int k = 1; k < 2 * d; k++)
            xpow[k] = xpow[k - 1] * xi;

        for (int r = 0; r < d; r++) {
            for (int c = 0; c < d; c++)
                XtX[r][c] += xpow[r + c];
            Xty[r] += xpow[r] * yi;
        }
    }

    /* build augmented matrix [XtX | Xty] */
    float aug[d][d + 1];
    for (int r = 0; r < d; r++) {
        for (int c = 0; c < d; c++)
            aug[r][c] = XtX[r][c];
        aug[r][d] = Xty[r];
    }

    /* forward elimination with partial pivoting */
    for (int col = 0; col < d; col++) {
        int max_row = col;
        float max_val = fabsf(aug[col][col]);
        for (int row = col + 1; row < d; row++) {
            if (fabsf(aug[row][col]) > max_val) {
                max_val = fabsf(aug[row][col]);
                max_row = row;
            }
        }
        if (max_val == 0.0f){
            return -1;
        }

        /* swap rows col and max_row */
        for (int k = 0; k <= d; k++) {
            float tmp = aug[col][k];
            aug[col][k] = aug[max_row][k];
            aug[max_row][k] = tmp;
        }

        for (int row = col + 1; row < d; row++) {
            float factor = aug[row][col] / aug[col][col];
            for (int k = col; k <= d; k++)
                aug[row][k] -= factor * aug[col][k];
        }
    }

    /* back substitution */
    float coeffs[d];
    for (int row = d - 1; row >= 0; row--) {
        coeffs[row] = aug[row][d];
        for (int k = row + 1; k < d; k++)
            coeffs[row] -= aug[row][k] * coeffs[k];
        coeffs[row] /= aug[row][row];
    }

    /* store coefficients in params modifying it in-place */
    for (int i = 0; i < d; i++)
        params[i] = coeffs[i];

    return 0;
}


/**
 * Collects inliers from points_x and points_y by computing the vertical
 * residual of each point from the polynomial model defined by params.
 * Points whose absolute residual is within threshold are written to
 * inliers_x and inliers_y in place, and n_inliers is incremented.
 *
 * For line models (n_params = 2) the vertical residual approximates
 * the perpendicular distance when the slope is small. For higher degree
 * models the vertical residual is the natural distance metric.
 *
 * Params:
 *     points_x    array of n_points floats
 *     points_y    array of n_points floats
 *     n_points    int, number of points to evaluate
 *     params      array of n_params floats, model coefficients
 *     n_params    int, number of model parameters
 *     threshold   float, maximum absolute residual to qualify as inlier
 *     inliers_x   array of floats, inliers written here in place
 *     inliers_y   array of floats, inliers written here in place
 *     n_inliers   int pointer, inlier count written here in place
 *
 * Returns:
 *     0  for success
 *     -1 for error if n_points < 1
 *     -1 for error if threshold <= 0
 *     -1 for error if n_params < 2
 */
int find_model_inliers(float* points_x, float* points_y, int n_points,
    float* params, int n_params, float threshold, float* inliers_x, 
    float* inliers_y, int* n_inliers)
{
    if (n_points < 1 || threshold <= 0 || n_params < 2) {
        return -1;
    }

    *n_inliers = 0; // initiate n_inliers
    for (int i = 0; i < n_points; i++) {
        float y_model = eval_model(points_x[i], params, n_params);
        if (fabs(points_y[i] - y_model) < threshold) {
            inliers_x[*n_inliers] = points_x[i];
            inliers_y[*n_inliers] = points_y[i];
            (*n_inliers)++;
        }
    }
    return 0;
}

/**
 * Combines inliers from two overlapping graphs of the same number of
 * parameters into a single refined polynomial model. Collects inliers
 * from each graph using their respective RANSAC-recovered models and
 * threshold via vertical residual, then refits one model to all
 * combined inliers using fit_model. This implements the stitching step
 * analogous to homography estimation — if both graphs share the same
 * underlying model, the combined fit is more accurate than either
 * individual fit since it uses more inlier points.
 *
 * Params:
 *     points_x1   array of n1 floats, x values of graph 1
 *     points_y1   array of n1 floats, y values of graph 1
 *     n1          int, number of points in graph 1
 *     params1     array of n_params floats, model for graph 1
 *     points_x2   array of n2 floats, x values of graph 2
 *     points_y2   array of n2 floats, y values of graph 2
 *     n2          int, number of points in graph 2
 *     params2     array of n_params floats, model for graph 2
 *     params      array of n_params floats, result stored here,
 *                 modified in place
 *     n_params    int, number of model parameters
 *     threshold   float, inlier vertical residual threshold
 *
 * Returns:
 *     int, total number of inliers used in the combined fit
 *     -1 for error if n1 < n_params
 *     -1 for error if n2 < n_params
 *     -1 for error if threshold <= 0
 *     -1 for error if no inliers found in either graph
 */
int stitch_models(float* points_x1, float* points_y1, int n1, float* params1,
                  float* points_x2, float* points_y2, int n2, float* params2,
                  float* params, int n_params, float threshold) {
    if (n1 < n_params || n2 < n_params || threshold < 0) {
        return -1;
    }
    // find inliers of model 1
    float inliers_x1[n1], inliers_y1[n1];
    int n_inliers1;
    int ret1 = find_model_inliers(points_x1, points_y1, n1, params1, n_params, 
        threshold, inliers_x1, inliers_y1, &n_inliers1);

    // find inliers of model 2
    float inliers_x2[n2], inliers_y2[n2];
    int n_inliers2;
    int ret2 = find_model_inliers(points_x2, points_y2, n2, params2, n_params, 
        threshold, inliers_x2, inliers_y2, &n_inliers2);

    // if either model has no inliers or find_model inliers does not work 
    // return error
    if (n_inliers1 < 1 || n_inliers2 < 1 || ret1 < 0 || ret2 < 0) {
        return -1;
    }
    
    // create an array of all inliers
    int n_inliers = n_inliers1 + n_inliers2;
    float inliers_x[n_inliers], inliers_y[n_inliers];
    for (int i = 0; i < n_inliers; i++) {
        if (i < n_inliers1) { // add inliers of model 1
            inliers_x[i] = inliers_x1[i];
            inliers_y[i] = inliers_y1[i];
        } else {  // add inliers of model 2
            inliers_x[i] = inliers_x2[i - n_inliers1];
            inliers_y[i] = inliers_y2[i - n_inliers1];
        }
    }

    // fills params in place
    int fit_ret = fit_model(inliers_x, inliers_y, n_inliers, params, n_params);
    if (fit_ret < 0) {
        printf("fit_ret = -1.");
        return -1;
    }
    return 0;
}


/**
 * Computes the perpendicular distance from each point to a line defined by 
 * slope and intercept, storing results in distances in place. This is used 
 * by ransac for line models (n_params = 2) where perpendicular distance is the 
 * natural metric. For higher degree models use find_model_inliers with 
 * vertical residual instead.
 *  distances[i] = |slope * points_x[i] - points_y[i] + intercept|
 *                  / sqrt(1 + slope^2)
 * Params:
 *      points_x    a list of n_points floats
 *      points_y    a list of n_points floats
 *      n_points    int
 *      slope       float
 *      intercept   float
 *      distances   a list of n_points floats, modified in place
 * Returns:
 *      0 for success
 * -1 for error if n_points <= 0
 */
int points_to_line_distances(float* points_x, float* points_y, int n_points,
                              float slope, float intercept, float* distances) {
    if (n_points <= 0)
        return -1;
    float denom = sqrtf(1 + slope * slope);
    for (int i = 0; i < n_points; i++)
        distances[i] = fabsf((slope * points_x[i] - points_y[i] + intercept)
                             / denom);
    return 0;
}


/**
 * Measures the Euclidean distance between the estimated and true polynomial
 * model parameters:
 *      sqrt(sum((params[i] - true_params[i])^2 for i in range(n_params)))
 * Params:
 *      params          a list of n_params floats, estimated coefficients
 *                      from lowest to highest degree [a0, a1, ...]
 *      true_params     a list of n_params floats, true coefficients
 *                      from lowest to highest degree [a0, a1, ...]
 *      n_params        int, number of model parameters
 * Returns:
 *      float, Euclidean distance between estimated and true parameters
 *      -1 for error if n_params < 2
 */
float model_error(float* params, float* true_params, int n_params) {
    if (n_params < 2)
        return -1.0f;
    float sum = 0.0f;
    for (int i = 0; i < n_params; i++)
        sum += (params[i] - true_params[i]) * (params[i] - true_params[i]);
    return sqrtf(sum);
}
