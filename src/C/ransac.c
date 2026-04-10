/*
Implementation of the RANSAC (Random Sample Consensus) algorithm for robust
polynomial model fitting in the presence of outliers.

This module provides the core ransac function and four helper functions for
estimating its parameters from the data. Rather than requiring the caller to
supply epsilon, k, d, and t directly, the helpers derive these values from
the residual distribution of a preliminary least squares fit.

Model parameters are stored as polynomial coefficients from lowest to highest
degree in a flat params array at offset pos * n_params:
    params[pos * n_params + 0] = a0  (intercept)
    params[pos * n_params + 1] = a1  (slope for line)
    params[pos * n_params + 2] = a2  (quadratic term)

For line fitting n_params = 2, giving y = a0 + a1 * x.

Functions:
    estimate_mean       computes population mean of distances
    estimate_std        computes population standard deviation of distances
    estimate_epsilon    estimates outlier fraction from residual distribution
    compute_t           estimates inlier threshold from residual distribution
    compute_k           computes required iterations from epsilon and n_params
    compute_d           computes expected inlier count from epsilon and n_points
    ransac              finds the best fitting model using RANSAC

References:
	For RANSAC:
    Fischler, M. A. and Bolles, R. C. 1981. Random sample consensus: a paradigm
    for model fitting with applications to image analysis and automated
    cartography. Commun. ACM 24, 6, 381-395.
	
	For Random Shuffle:
    Durstenfeld, R. 1964. Algorithm 235: Random permutation.
    Communications of the ACM 7, 7, 420.
*/

