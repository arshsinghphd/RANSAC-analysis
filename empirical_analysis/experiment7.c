/* =============================================================================
 * experiments7.c
 *
 * Experiment 7: How does RANSAC performance scale with dataset size N?
 *
 *     Fixes epsilon = 0.3 and linear model (m = 2). Varies N from 2
 *     to 8192 doubling each step. k is computed from true epsilon for
 *     each run. Uses malloc for point arrays to avoid stack overflow
 *     at large N. Repeats N_REPEATS times per condition.
 *
 * Output:
 *     results/exp7.csv
 *     columns: index, n, epsilon, d, m, k, t,
 *              repeat, time_mu_s, model_error
 *
 * Usage:
 *     make exp7
 * ========================================================================== */


#include "experiments.h"

#define EXP6_CSV  "results/exp6.csv"
#define DEGREE_MIN 2
#define DEGREE_MAX 5
#define DEGREE_STEPS (DEGREE_MAX - DEGREE_MIN + 1)
#define T_FIXED 1  // threshold fixed at 2 * NOISE_STD
#define EPS_FIXED  0.15f
#define N_BUDGETS  2

static const int K_BUDGETS[N_BUDGETS] = {800, 8000};

int main(void) {
    srand((unsigned int) time(NULL));

    /* open output file */
    FILE* fp = fopen(EXP6_CSV, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s\n", EXP6_CSV);
        return 1;
    }

    /* CSV header */
    fprintf(fp, "index,n,epsilon,d,m,k_budget,k_theory,t," 
            "repeat,time_mu_s,model_error\n");

    int n_inliers = (int)((1.0f - EPS_FIXED) * N_TOTAL);
    int n_outliers = N_TOTAL - n_inliers;
    int d = compute_d(EPS_FIXED, N_TOTAL);
    int index  = 0;

    for (int b = 0; b < N_BUDGETS; b++) {
        int k_budget = K_BUDGETS[b];

        printf("\n--- k_budget = %d ---\n", k_budget);

        for (int deg = DEGREE_MIN; deg <= DEGREE_MAX; deg++) {
            int n_params = deg + 1;

            /* true params [1, 1, ..., 1] for this degree */
            float true_params[n_params];
            for (int j = 0; j < n_params; j++)
            true_params[j] = 1.0f;

            /* theoretical k for this degree at epsilon = 0.5 */
            int k_theory = compute_k(EPS_FIXED, n_params, FAIL_PROB);

            for (int r = 0; r < N_REPEATS; r++) {
                float points_x[N_TOTAL], points_y[N_TOTAL];
                float t;

                /* generate noisy data with outliers, no structural bias. */
                make_data(points_x, points_y,
                            n_inliers, n_outliers,
                            true_params, n_params,
                            NOISE_STD, &t,
                            0, NULL, 0.0f);

                /* run ransac with fixed k_budget, and threshold T_FIXED. */
                RansacResult res = run_ransac(points_x, points_y,
                                                N_TOTAL, n_params,
                                                true_params,
                                                EPS_FIXED, T_FIXED, d, k_budget,
                                                r, index++);

                /* write one CSV row */
                fprintf(fp, "%d,%d,%.2f,%d,%d,%d,%d,%.4f,%d,%.2f,%.6f\n",
                        res.index, res.n, res.epsilon,
                        res.d, res.m, k_budget, k_theory, res.t,
                        res.repeat, res.time_mu_s, res.model_error);
            }
        }
    }
    fclose(fp);
    printf("\nExperiment 6 done. Results written to %s\n", EXP6_CSV);
    return 0;
}