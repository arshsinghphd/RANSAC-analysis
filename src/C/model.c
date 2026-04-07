/* Implementation of model.h. */

#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

/**
 * Fits a polynomial model of degree (n_params - 1) to n_points data points 
 * using least squares, solved via Gaussian elimination on the normal equations. 
 * 
 * Stores the n_params coefficients in params starting at index pos * n_params, 
 * in order from lowest to highest degree:
 *  params[pos * n_params + 0] = a0  (intercept) 
 *  params[pos * n_params + 1] = a1  (slope for line)
 *  params[pos * n_params + 2] = a2  (quadratic term)
 *  ...
 * 
 * For n_params = 2 this is equivalent to ordinary least squares (OLS):
 *  y = a0 + a1 * x
 * 
 * The normal equations (X^T X) a = X^T y are formed using the Vandermonde
 * matrix X and solved using Gaussian elimination with partial pivoting and
 * back substitution.
 * 
 * Runs in O(n_points) for n_params << n_points.
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
int fit_model(float* points_x, float* points_y, int n_points,
              float* params, int n_params, int pos) {
    if (n_points < n_params || n_params < 2 || pos < 0)
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

    /* store coefficients in params at offset pos * n_params */
    int offset = pos * n_params;
    for (int i = 0; i < d; i++)
        params[offset + i] = coeffs[i];

    return 0;
}





