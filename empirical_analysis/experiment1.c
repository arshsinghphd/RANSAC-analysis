/* =============================================================================
 * experiment1.c
 *
 * Experiment 1: Does wall-clock time grow linearly with k?
 *
 *     Fixes epsilon = 0.5. Varies k from 10 to 500 in steps of 10.
 *     Runs for linear (m = 2) and quadratic (m = 3) models.
 *     N_TOTAL = 1000 is fixed throughout.
 *     Repeats N_REPEATS times per condition.
 *
 * Output:
 *     results/exp1.csv
 *     columns: index, n, epsilon, d, m, k, t, repeat, time_mu_s, model_error
 *
 * Usage:
 *     make exp1
 * ========================================================================== */

#include "experiments.h"

#define EXP1_CSV "results/exp1.csv"
// k sweep range — 10 to 500 in steps of 10
#define K_MIN 10
#define K_MAX 500
#define K_STEP 10
// total number of k values: (500 - 10) / 10 + 1 = 50
#define K_STEPS ((K_MAX - K_MIN) / K_STEP + 1)
// two models: linear (m=2) and quadratic (m=3)
#define N_MODELS 2
// outlier fraction fixed for all runs — k is the only variable
#define EPS_FIXED 0.5f


int main(void) {
    srand((unsigned int) time(NULL));

    // open csv for writing — created fresh each run
    FILE* fp = fopen(EXP1_CSV, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s\n", EXP1_CSV);
        return 1;
    }

    // write header
    fprintf(fp, "index,n,epsilon,d,m,k,t,repeat,time_mu_s,model_error\n");

    // model configurations: linear (2 params) and quadratic (3 params)
    int n_params_list[N_MODELS] = {2, 3};
    float true_params_linear[] = _TRUE_PARAMS_LINEAR;
    float true_params_quadratic[] = _TRUE_PARAMS_QUADRATIC;
    const float* true_params_list[N_MODELS] = {
        true_params_linear,
        true_params_quadratic
    };

    // n_inliers and n_outliers fixed since epsilon is fixed
    int n_inliers = (int)((1.0f - EPS_FIXED) * N_TOTAL);
    int n_outliers = N_TOTAL - n_inliers;
    // d is the minimum consensus size — derived from epsilon
    int d = compute_d(EPS_FIXED, N_TOTAL);
    int index = 0;

    for (int m = 0; m < N_MODELS; m++) {
        int n_params = n_params_list[m];
        const float* true_params = true_params_list[m];

        printf("\n--- m=%d ---\n", n_params);

        // vary k from K_MIN to K_MAX — this is the independent variable
        for (int ki = 0; ki < K_STEPS; ki++) {
            int k = K_MIN + ki * K_STEP;

            // repeat each (m, k) condition to get stable time estimates
            for (int r = 0; r < N_REPEATS; r++) {
                float points_x[N_TOTAL], points_y[N_TOTAL];
                // t computed from data by make_data — not controlled here
                float t;

                // generate fresh noisy data with outliers for each repeat
                make_data(points_x, points_y,
                          n_inliers, n_outliers,
                          true_params, n_params,
                          NOISE_STD, &t,
                          0, NULL, 0.0f);

                // run ransac and record wall-clock time inside run_ransac
                RansacResult res = run_ransac(points_x, points_y,
                                              N_TOTAL, n_params,
                                              true_params,
                                              EPS_FIXED, t, d, k,
                                              r, index++);

                // write one row per repeat
                fprintf(fp, "%d,%d,%.2f,%d,%d,%d,%.4f,%d,%.2f,%.6f\n",
                        res.index, res.n, res.epsilon,
                        res.d, res.m, k, res.t,
                        res.repeat, res.time_mu_s, res.model_error);
            }
        }
    }

    fclose(fp);
    printf("\nExperiment 1 done. Results written to %s\n", EXP1_CSV);
    return 0;
}