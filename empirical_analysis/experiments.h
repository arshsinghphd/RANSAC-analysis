/* =============================================================================
 * experiments.h
 *
 * Shared constants, bias function declarations, and function declarations
 * for the RANSAC empirical analysis. Included by experiment1.c,
 * experiment2.c, and experiment3.c.
 *
 * All experiments use synthetic two-dimensional Cartesian point sets
 * generated from known polynomial models. Total dataset size N_TOTAL
 * is fixed across all experiments to isolate the effect of the variable
 * under study.
 * ========================================================================== */

#ifndef EXPERIMENTS_H
#define EXPERIMENTS_H

#include "../src/generator.h"
#include "../src/model.h"
#include "../src/ransac.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ---------------------------------------------------------------
 * Shared experiment constants
 * --------------------------------------------------------------- */
#define N_TOTAL     1000        /* total points per run, always fixed */
#define X_MIN       0.0f        /* lower bound of x range             */
#define X_MAX       999.0f      /* upper bound of x range             */
#define NOISE_STD   0.5f        /* gaussian noise standard deviation  */
#define FAIL_PROB   0.01f       /* RANSAC failure probability         */
#define N_REPEATS   100          /* independent runs per condition     */

/* ---------------------------------------------------------------
 * True model parameters — lowest to highest degree.
 *     linear:    y = 5 + 2x
 *     quadratic: y = 1 + x + x^2
 * --------------------------------------------------------------- */
#define _TRUE_PARAMS_LINEAR    {5.0f, 2.0f}
#define _TRUE_PARAMS_QUADRATIC {1.0f, 1.0f, 1.0f}

/* ---------------------------------------------------------------
 * Result struct — maps one RANSAC run to one CSV row.
 * Fields correspond directly to CSV columns:
 *     index, n, epsilon, t, d, m, k, repeat, time_mu_s, model_error
 * --------------------------------------------------------------- */
typedef struct {
    int   index;
    int   n;
    int   m;
    float epsilon;
    float t;
    int   d;
    int   k;
    int   repeat;
    float time_mu_s;
    float model_error;
} RansacResult;

/* ---------------------------------------------------------------
 * Bias functions for structural bias experiments.
 * Each takes x and returns a bias value to add to y.
 * --------------------------------------------------------------- */
float bias_constant(float x);
float bias_linear_bias(float x);
float bias_periodic(float x);

/* ---------------------------------------------------------------
 * Generates noisy inlier data, optionally applies structural bias,
 * and appends outliers. Fills points_x and points_y in place.
 * Returns threshold t via pointer.
 *
 * Params:
 *     points_x        array of N_TOTAL floats, modified in place
 *     points_y        array of N_TOTAL floats, modified in place
 *     n_inliers       int, number of inlier points
 *     n_outliers      int, number of outlier points
 *     true_params     array of n_params floats, true model coefficients
 *     n_params        int, number of model parameters
 *     noise_std       float, gaussian noise std; pass 0 for clean data
 *     t_out           float pointer, estimated threshold written here
 *     apply_bias      int, 1 to apply structural bias, 0 to skip
 *     bias_fn         float (*)(float), bias function, ignored if
 *                     apply_bias == 0 or NULL
 *     pr              float, fraction of inliers to bias in [0, 1]
 * --------------------------------------------------------------- */
void make_data(float* points_x, float* points_y,
                int n_inliers, int n_outliers,
                const float* true_params, int n_params,
                float noise_std,
                float* t_out,
                int apply_bias,
                float (*bias_fn)(float), float pr);

/* ---------------------------------------------------------------
 * Runs ransac, times it, and returns a RansacResult struct.
 * model_error is -1.0 if ransac fails.
 *
 * Params:
 *     points_x        array of n_points floats
 *     points_y        array of n_points floats
 *     n_points        int, total number of points
 *     n_params        int, number of model parameters
 *     true_params     array of n_params floats, true model coefficients
 *     epsilon         float, outlier fraction
 *     t               float, inlier threshold
 *     d               int, expected inlier count
 *     k               int, number of iterations
 *     repeat          int, repeat index for this run
 *     index           int, global row index for CSV
 * --------------------------------------------------------------- */
RansacResult run_ransac(float* points_x, float* points_y,
                        int n_points, int n_params,
                        const float* true_params,
                        float epsilon, float t, int d, int k,
                        int repeat, int index);

#endif /* EXPERIMENTS_H */