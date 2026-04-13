/* =============================================================================
 * experiments.c implementation of experiment.h
 *
 * Shared data generation and RANSAC runner for the
 * empirical analysis. See experiments.h for declarations.
 * ========================================================================== */

#include "experiments.h"

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


/* ---------------------------------------------------------------
 * run_ransac
 * --------------------------------------------------------------- */
RansacResult run_ransac(float* points_x, float* points_y,
                         int n_points, int n_params,
                         const float* true_params,
                         float epsilon, float t, int d, int k,
                         int repeat, int index) {
    RansacResult res;
    res.index       = index;
    res.n           = n_points;
    res.m           = n_params;
    res.epsilon     = epsilon;
    res.t           = t;
    res.d           = d;
    res.k           = k;
    res.repeat      = repeat;
    res.model_error = -1.0f;
    res.time_mu_s     = 0.0f;

    float return_array[2 + n_params];

    /* time the ransac call */
    clock_t start = clock();
    int ret = ransac(points_x, points_y, n_points, n_params,
                     k, t, d, return_array);
    clock_t end = clock();

    res.time_mu_s = (float)(end - start) / CLOCKS_PER_SEC * 1e6f;

    if (ret == -1)
        return res;

    /* compute model error */
    float est_params[n_params];
    float true_p[n_params];
    for (int j = 0; j < n_params; j++) {
        est_params[j] = return_array[2 + j];
        true_p[j]     = true_params[j];
    }
    res.model_error = model_error(est_params, true_p, n_params);

    return res;
}


/* Development tests */

// int main(void) {
//     srand((unsigned int) time(NULL));

//     printf("experiments.c sanity check\n");
//     printf("---------------------------\n");

//     /* linear model, epsilon = 0.3 */
//     float true_params[] = _TRUE_PARAMS_LINEAR;
//     int   n_params  = 2;
//     float epsilon   = 0.3f;
//     int   n_inliers = (int)((1.0f - epsilon) * N_TOTAL);
//     int   n_outliers = N_TOTAL - n_inliers;
//     int   d         = compute_d(epsilon, N_TOTAL);
//     int   k         = compute_k(epsilon, n_params, FAIL_PROB);

//     float points_x[N_TOTAL], points_y[N_TOTAL];
//     float t;

//     make_data(points_x, points_y, n_inliers, n_outliers,
//               true_params, n_params, NOISE_STD, &t,
//               0, NULL, 0.0f);

//     printf("n_inliers  = %d\n", n_inliers);
//     printf("n_outliers = %d\n", n_outliers);
//     printf("t          = %.4f\n", t);
//     printf("d          = %d\n", d);
//     printf("k          = %d\n", k);

//     RansacResult res = run_ransac(points_x, points_y, N_TOTAL,
//                                   n_params, true_params,
//                                   epsilon, t, d, k, 0, 0);

//     printf("model_error = %.6f\n", res.model_error);
//     printf("time_mu_s     = %.2f\n", res.time_mu_s);

//     return 0;
// }
