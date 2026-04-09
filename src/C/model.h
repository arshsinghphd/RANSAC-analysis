/**
 * Model fitting functions for RANSAC pipeline. Supports polynomial models of
 * any degree via least squares solved using Gaussian elimination on the normal
 * equations. Provides stitching of two overlapping graphs into a single refined
 * model, analogous to image stitching in homography estimation.
 * 
 * Model parameters are stored as polynomial coefficients from lowest to highest
 * degree:
 * params[pos * n_params + 0] = a0  (intercept / constant term)
 * params[pos * n_params + 1] = a1  (slope / linear term)
 * params[pos * n_params + 2] = a2  (quadratic term)
 * ...
 * For line fitting n_params = 2, giving y = a0 + a1 * x.
 * For quadratic fitting n_params = 3, giving y = a0 + a1 * x + a2 * x^2.
 * 
 * Functions:
 * eval_model                  evaluates polynomial model at a given x
 * fit_model                   fits polynomial model using least squares
 * find_model_inliers          collects inliers using vertical residual
 * stitch_models               combines inliers from two graphs into one model
 * points_to_line_distances    computes perpendicular distances to a line
 * model_error                 measures Euclidean distance between two models
 * 
 * References:
 * Fischler, M. A. and Bolles, R. C. 1981. Random sample consensus: a paradigm
 * for model fitting with applications to image analysis and automated
 * cartography. Commun. ACM 24, 6, 381-395.
 */

/**
 * Fits a polynomial model of degree n_params - 1 to n_points data points using
 * least squares, solved via Gaussian elimination on the normal equations. 
 * Stores the n_params coefficients in params starting at index pos * n_params, 
 * in order from lowest to highest degree:
 * params[pos * n_params + 0] = a0  (intercept) 
 * params[pos * n_params + 1] = a1  (slope for line)
 * params[pos * n_params + 2] = a2  (quadratic term)
 * ...
 * For n_params = 2 this is equivalent to ordinary least squares line fitting:
 * y = a0 + a1 * x
 * The normal equations (X^T X) a = X^T y are formed using the Vandermonde
 * matrix X and solved using Gaussian elimination with partial pivoting and
 * back substitution. All matrix operations are implemented without external
 * libraries for C-correspondence.
 * 
 * Simple introduction by Robert Cappetta in this video on YouTube: 
 *  https://youtu.be/8cnxU-Pmb3w
 * Cappetta, R. 2018. Gaussian Elimination with Back Substitution. YouTube. 
 * Retrieved from https://youtu.be/8cnxU-Pmb3w on Apr 7, 2026.
 * 
 * Params:
 * points_x     a list of n_points floats
 * points_y     a list of n_points floats
 * n_points     int, number of points to fit
 * params       a list of floats of size at least (pos + 1) * n_params, 
 *              modified in place
 * n_params     int, number of model parameters, equals polynomial degree + 1
 * 
 * Returns:
 * 0 for success
 * -1 for error if n_points < n_params
 * -1 for error if n_params < 2-1 for error if pos < 0
 * -1 for error if matrix is singular (e.g. all x values equal)
 */
int fit_model(float* points_x, float* points_y, int n_points, float* params, 
    int n_params);

/**
 * Collects inliers from points_x and points_y by computing the vertical
 * residual of each point from the polynomial model defined by params.
 * Points whose absolute residual is within threshold are appended to
 * inliers_x and inliers_y in place.
 * 
 * For line models (n_params = 2) the vertical residual approximates the
 * perpendicular distance when the slope is small. For higher degree models
 * the vertical residual is the natural distance metric.
 * 
 * Params:
 * points_x    a list of n_points floats
 * points_y    a list of n_points floats
 * n_points    int, number of points to evaluate
 * params      a list of floats containing model coefficients
 * n_params    int, number of model parameters
 * threshold   float, maximum absolute residual to qualify as inlier
 * inliers_x   a list of floats, appended to in place
 * inliers_y   a list of floats, appended to in place
 * 
 * Returns:
 * 0 for success
 * -1 for error if n_points < 1
 * -1 for error if threshold <= 0
 * -1 for error if n_params < 2
 * -1 for error if pos < 0
 */
int find_model_inliers(float* points_x, float* points_y, int n_points,
    float* params, int n_params, float threshold, float* inliers_x, 
    float* inliers_y):
