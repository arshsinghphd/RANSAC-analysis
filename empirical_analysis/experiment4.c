/* =============================================================================
 * experiments4.c
 *
 * Experiment 4: How does RANSAC break down as outlier fraction increases?
 *
 *     Fixes k at the value computed for epsilon = 0.5 and varies epsilon
 *     from 0.1 to 0.9 in steps of 0.1. Runs for linear (m = 2) and
 *     quadratic (m = 3) models. N_TOTAL = 1000 is fixed throughout.
 *     Repeats N_REPEATS times per condition.
 *
 * Output:
 *     results/exp4.csv
 *     columns: index, n, epsilon, d, m, k, t, repeat, time_mu_s, model_error
 *
 * Usage:
 *     make exp4
 * ========================================================================== */

#include "experiments.h"

#define EXP4_CSV      "results/exp4.csv"
#define EPSILON_STEPS 19
#define N_MODELS      2

static const float EPSILONS[EPSILON_STEPS] = {
    0.05f, 0.1f, 0.15f, 0.2f, 0.25f, 0.3f, 0.35f, 0.4f, 0.45f, 0.5f, 0.55f, 
    0.6f, 0.65f, 0.7f, 0.75f, 0.8f, 0.85f, 0.9f, 0.95f
};


int main(void) {
    srand((unsigned int) time(NULL));

    /* open output file */
    FILE* fp = fopen(EXP4_CSV, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s\n", EXP4_CSV);
        return 1;
    }

    /* CSV header */
    fprintf(fp, "index,n,epsilon,d,m,k,t,repeat,time_mu_s,model_error\n");

    /* model configurations */
    int n_params_list[N_MODELS] = {2, 3};
    float true_params_linear[]    = _TRUE_PARAMS_LINEAR;
    float true_params_quadratic[] = _TRUE_PARAMS_QUADRATIC;
    const float* true_params_list[N_MODELS] = {
        true_params_linear,
        true_params_quadratic
    };

    int index = 0;

    for (int m = 0; m < N_MODELS; m++) {
        int          n_params    = n_params_list[m];
        const float* true_params = true_params_list[m];

        /* fix k at epsilon = 0.5 for the entire experiment */
        int k_fixed = compute_k(0.5f, n_params, FAIL_PROB);

        printf("model m=%d  k_fixed=%d\n", n_params, k_fixed);

        for (int e = 0; e < EPSILON_STEPS; e++) {
            float epsilon    = EPSILONS[e];
            int   n_inliers  = (int)((1.0f - epsilon) * N_TOTAL);
            int   n_outliers = N_TOTAL - n_inliers;
            int   d          = compute_d(epsilon, N_TOTAL);

            for (int r = 0; r < N_REPEATS; r++) {
                float points_x[N_TOTAL], points_y[N_TOTAL];
                float t;

                /* generate noisy data with outliers, no structural bias */
                make_data(points_x, points_y,
                          n_inliers, n_outliers,
                          true_params, n_params,
                          NOISE_STD, &t,
                          0, NULL, 0.0f);

                /* run ransac and collect result */
                RansacResult res = run_ransac(points_x, points_y,
                                              N_TOTAL, n_params,
                                              true_params,
                                              epsilon, t, d, k_fixed,
                                              r, index++);

                /* write one CSV row */
                fprintf(fp, "%d,%d,%.2f,%d,%d,%d,%.4f,%d,%.2f,%.6f\n",
                        res.index, res.n, res.epsilon, res.d, 
                        res.m, res.k, res.t, res.repeat,
                        res.time_mu_s, res.model_error);
            }
        }
    }
    fclose(fp);
    printf("Experiment 4 done. Results written to %s\n", EXP4_CSV);
    return 0;
}