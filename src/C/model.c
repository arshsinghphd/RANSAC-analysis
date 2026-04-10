/* Implementation of model.h. */

#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

/**
Evaluates the polynomial model at a given x using Horner's method.
The polynomial is defined by n_params coefficients stored in params
starting at offset pos n_params, from lowest to highest degree:

    y = a0 + a1 x + a2 x^2 + ... + a(n-1) x^(n-1)

Params:
    x           float, the x value to evaluate at
    params      a list of floats containing model coefficients
    n_params    int, number of coefficients
    pos         int, position index, coefficients at pos n_params

Returns:
    float, the model value at x
    -1 for error if n_params < 2
    -1 for error if pos < 0
 */
float eval_model(float x, float* params, int n_params) {
    if (n_params < 2)
        return -1;
    float y = params[0];
    for(int i = 1; i < n_params; i++)
        y += params[i] * pow(x, i);
    return y;
}


/**
Fits a polynomial model of degree (n_params - 1) to n_points data points 
using least squares, solved via Gaussian elimination on the normal equations. 

Stores the n_params coefficients in params starting at index pos n_params, 
in order from lowest to highest degree:
    params[pos n_params + 0] = a0  (intercept) 
    params[pos n_params + 1] = a1  (slope for line)
    params[pos n_params + 2] = a2  (quadratic term)
    ...

For n_params = 2 this is equivalent to ordinary least squares (OLS):
    y = a0 + a1 x

The normal equations (X^T X) a = X^T y are formed using the Vandermonde
matrix X and solved using Gaussian elimination with partial pivoting and
back substitution.

Runs in O(n_points) for n_params << n_points.

Params:
    points_x     a list of n_points floats
    points_y     a list of n_points floats
    n_points     int, number of points to fit
    params       a list of floats of size at least (pos + 1) n_params, 
              modified in place
    n_params     int, number of model parameters, equals polynomial degree + 1

Returns:
    0 for success
    -1 for error if n_points < n_params
    -1 for error if n_params < 2-1 for error if pos < 0
    -1 for error if matrix is singular (e.g. all x values equal)
 */
int fit_model(float* points_x, float* points_y, int n_points,
              float* params, int n_params) {
    if (n_points < n_params || n_params < 2)
        return -1;

    // test if x_min == x_max
    float x_min = points_x[0];
    float x_max = points_x[0];
    for (int i = 0; i < n_points; i++) {
        if (points_x[i] < x_min)
            x_min = points_x[i];
        if (points_x[i] > x_max)
            x_max = points_x[i];
    }
    if (x_min - x_max < 1e-6) // x_min == x_max with floats
        return -1;

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
        if (max_val == 0.0f)
            return -1;

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
Collects inliers from points_x and points_y by computing the vertical
residual of each point from the polynomial model defined by params.
Points whose absolute residual is within threshold are appended to
inliers_x and inliers_y in place.

For line models (n_params = 2) the vertical residual approximates the
perpendicular distance when the slope is small. For higher degree models
the vertical residual is the natural distance metric.

Params:
    points_x    a list of n_points floats
    points_y    a list of n_points floats
    n_points    int, number of points to evaluate
    params      a list of floats containing model coefficients
    n_params    int, number of model parameters
    threshold   float, maximum absolute residual to qualify as inlier
    inliers_x   a list of floats, appended to in place
    inliers_y   a list of floats, appended to in place
    n_inliers   a pointer to count of inliers, modified in place

Returns:
    0 for success
    -1 for error if n_points < 1
    -1 for error if threshold <= 0
    -1 for error if n_params < 2
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

/*
Combines inliers from two overlapping graphs into a single refined
polynomial model. Collects inliers from each graph using their respective
RANSAC-recovered models and threshold via vertical residual, then refits
one model to all combined inliers using fit_model. This implements the
stitching step analogous to homography estimation — if both graphs share
the same underlying model, the combined fit is more accurate than either
individual fit since it uses more inlier points.

Params:
    points_x1   a list of n1 floats, x values of graph 1
    points_y1   a list of n1 floats, y values of graph 1
    n1          int, number of points in graph 1
    params1     a list of n_params floats, model for graph 1 at pos 0
    points_x2   a list of n2 floats, x values of graph 2
    points_y2   a list of n2 floats, y values of graph 2
    n2          int, number of points in graph 2
    params2     a list of n_params floats, model for graph 2 at pos 0
    n_params    int, number of model parameters
    threshold   float, inlier residual threshold
    params      a list of floats of size at least (pos + 1) * n_params,
                modified in place

Returns:
    int, total number of inliers used in the combined fit
    -1 for error if n1 < n_params
    -1 for error if n2 < n_params
    -1 for error if threshold <= 0
    -1 for error if pos < 0
    -1 for error if no inliers found in either graph
*/
int stitch_models(points_x1, points_y1, n1, params1,
                  points_x2, points_y2, n2, params2,
                  params, n_params, threshold) {
    if (n1 < n_params || n2 < n_params || threshold <= 0)
        return -1;
    // find inliers of model 1
    float inliers_x1[n1], inliers_y1[n1];
    int n_inliers1;
    find_model_inliers(points_x1, points_y1, n1, params1, n_params, threshold, 
        inliers_x1, inliers_y1, &n_inliers1);
    // find inliers of model 2
    float inliers_x1[n2], inliers_y1[n2];
    int n_inliers2;
    find_model_inliers(points_x2, points_y2, n2, params2, n_params, threshold, 
        inliers_x2, inliers_y2, &n_inliers2);
    
    // if either model has no inliers return error
    if (n_inliers1 < 1 || n_inliers2 < 1)
        return -1
    
    // create an array of all inliers
    int n_inliers = n_inliers1 + n_inliers2;
    float inliers_x[n_inliers];
    for (int i = 0; i < n_inliers; i++) {
        if (i < n_inliers1) {. // add inliers of model 1
            inliers_x[i] = inliers_x1[i];
            inliers_y[i] = inliers_y1[i];
        } else {  // add inliers of model 2
            inliers_x[i] = inliers_x2[i - n_inliers1];
            inliers_y[i] = inliers_y2[i - n_inliers1];
        }
    }
    // fills params in place
    fit_model(inliers_x, inliers_y, n_inliers, params, n_params);
    return 0;
}