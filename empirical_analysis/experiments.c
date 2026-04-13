/* =============================================================================
 * experiments.c implementation of experiment.h
 *
 * Shared data generation and RANSAC runner for the
 * empirical analysis. See experiments.h for declarations.
 * ========================================================================== */

#include "experiments.h"

/* -----------------------------------------------------------------------------
 * True model parameters
----------------------------------------------------------------------------- */
const float TRUE_PARAMS_LINEAR[]    = TRUE_PARAMS_LINEAR;
const float TRUE_PARAMS_QUADRATIC[] = TRUE_PARAMS_QUADRATIC;


/* -----------------------------------------------------------------------------
 * Bias functions
----------------------------------------------------------------------------- */

float bias_constant(float x) {
    (void) x;
    return 1.0f;
}

float bias_linear_bias(float x) {
    return 0.5f * x;
}

float bias_periodic(float x) {
    return sinf(x);
}


/* ---------------------------------------------------------------
 * make_data
 * --------------------------------------------------------------- */
void make_data(float* points_x, float* points_y,
               int n_inliers, int n_outliers,
               const float* true_params, int n_params,
               float noise_std,
               float* t_out,
               int apply_bias,
               float (*bias_fn)(float), float pr) {

    /* copy true_params into a mutable local array */
    float params[n_params];
    for (int j = 0; j < n_params; j++)
        params[j] = true_params[j];

    /* generate inliers on the true polynomial model */
    make_inliers(points_x, points_y, n_inliers,
                 params, n_params, X_MIN, X_MAX);

    /* add gaussian noise if noise_std > 0, else clean data */
    if (noise_std > 0.0f)
        add_gaussian_noise(points_y, n_inliers, noise_std);

    /* optionally apply structural bias to fraction pr of inliers */
    if (apply_bias && bias_fn != NULL && pr > 0.0f)
        add_structural_bias(points_y, points_x, n_inliers, pr, bias_fn);

    /* estimate threshold from inlier data before outliers are added */
    float t = compute_t(points_x, points_y, n_inliers, n_params);
    if (t < 1e-6f)
        t = 2.0f * noise_std
            * sqrtf(1.0f + true_params[1] * true_params[1]);
    if (t < 1e-6f)
        t = 1e-4f;
    *t_out = t;

    /* append outliers after threshold is estimated */
    if (n_outliers > 0)
        add_outliers(points_x, points_y, n_inliers, n_outliers,
                     params, n_params, noise_std > 0.0f ? noise_std
                                                        : NOISE_STD);
}
