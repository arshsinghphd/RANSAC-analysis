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
 * fit_model
 * 
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

    // check all x values are not identical — singular matrix otherwise
    float x_min = points_x[0];
    float x_max = points_x[0];
    for (int i = 0; i < n_points; i++) {
        if (points_x[i] < x_min) x_min = points_x[i];
        if (points_x[i] > x_max) x_max = points_x[i];
    }
    // 1e-4 tolerance accounts for float rounding near equal values
    if (fabsf(x_min - x_max) < 1e-4)
        return -1;

    // m is the number of polynomial coefficients — also the matrix dimension
    int m = n_params;

    // XtX is the m x m Gram matrix: XtX[r][c] = sum of x^(r+c) over all points
    // Xty is the m-vector: Xty[r] = sum of x^r * y over all points
    // use double throughout to reduce numerical error on ill-conditioned Vandermonde
    double XtX[m][m];
    double Xty[m];
    for (int i = 0; i < m; i++) {
        Xty[i] = 0.0;
        for (int j = 0; j < m; j++)
            XtX[i][j] = 0.0;
    }

    for (int i = 0; i < n_points; i++) {
        double xi = (double) points_x[i];
        double yi = (double) points_y[i];

        // precompute powers of xi up to x^(2m-1) — needed for both rows and cols of XtX
        double xpow[2 * m];
        xpow[0] = 1.0;
        for (int k = 1; k < 2 * m; k++)
            xpow[k] = xpow[k - 1] * xi;

        // accumulate XtX[r][c] += xi^(r+c) and Xty[r] += xi^r * yi
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < m; c++)
                XtX[r][c] += xpow[r + c];
            Xty[r] += xpow[r] * yi;
        }
    }

    // build augmented matrix [XtX | Xty] for Gaussian elimination
    // aug[r][m] holds the right-hand side Xty[r]
    double aug[m][m + 1];
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < m; c++)
            aug[r][c] = XtX[r][c];
        aug[r][m] = Xty[r];
    }

    // forward elimination with partial pivoting
    // partial pivoting swaps the row with the largest pivot to the top
    // to reduce numerical error from dividing by small values
    for (int col = 0; col < m; col++) {
        int max_row = col;
        double max_val = fabs(aug[col][col]);
        for (int row = col + 1; row < m; row++) {
            if (fabs(aug[row][col]) > max_val) {
                max_val = fabs(aug[row][col]);
                max_row = row;
            }
        }
        // singular matrix — no unique solution
        if (max_val == 0.0)
            return -1;

        // swap current row with the pivot row
        for (int k = 0; k <= m; k++) {
            double tmp = aug[col][k];
            aug[col][k] = aug[max_row][k];
            aug[max_row][k] = tmp;
        }

        // eliminate all entries below the pivot in this column
        // factor is the multiplier that zeros out aug[row][col]
        for (int row = col + 1; row < m; row++) {
            double factor = aug[row][col] / aug[col][col];
            for (int k = col; k <= m; k++)
                aug[row][k] -= factor * aug[col][k];
        }
    }

    // back substitution — solve upper triangular system from bottom to top
    // each coefficient is solved using the already-known coefficients below it
    double coeffs[m];
    for (int row = m - 1; row >= 0; row--) {
        coeffs[row] = aug[row][m];
        for (int k = row + 1; k < m; k++)
            coeffs[row] -= aug[row][k] * coeffs[k];
        // divide by the diagonal element to isolate coeffs[row]
        coeffs[row] /= aug[row][row];
    }

    // cast back to float — precision was only needed internally
    for (int i = 0; i < m; i++)
        params[i] = (float) coeffs[i];

    return 0;
}


/**
 * find_model_inliers
 * 
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
        // vertical residual < threshold
        if (fabs(points_y[i] - y_model) < threshold) {
            inliers_x[*n_inliers] = points_x[i];
            inliers_y[*n_inliers] = points_y[i];
            (*n_inliers)++;
        }
    }
    return 0;
}

/**
 * stitch_models
 * 
 * Combines inliers from two overlapping graphs of the same number of
 * parameters into a single refined polynomial model. Collects inliers
 * from each graph using their respective RANSAC-recovered models and
 * threshold via vertical residual, then refits one model to all
 * combined inliers using fit_model. This implements the stitching step
 * analogous to homography estimation — if both graphs share the same
 * underlying model, the combined fit is more accurate than either
 * individual fit since it uses more inlier points.
 * 
 * Used to created motivating example for the report.
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
 * points_to_line_distances
 * 
 * Computes the perpendicular distance from each point to a line defined by 
 * slope and intercept, storing results in distances in place. This could be 
 * used for line models (n_params = 2) where perpendicular distance is the 
 * natural metric. For higher degree models use find_model_inliers with 
 * vertical residual instead.
 * 
 *      distances[i] = |slope * points_x[i] - points_y[i] + intercept|
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
 * model_error
 * 
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
