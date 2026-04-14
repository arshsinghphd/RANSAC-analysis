/* =============================================================================
 * experiments3.c
 *
 * Experiment 3: At what model degree does RANSAC fail for a fixed
 * iteration budget?
 *
 *  Fixes epsilon = 0.5. Varies polynomial degree from 2 to 10
 *      (n_params = 3 to 11). 
 *  Runs twice: once with k = 100 and once with k = 1000, 
 *      to show that the budget drives breakdown.
 *  True params are [1, 1, ..., 1] for all degrees.
 *  N_TOTAL = 1000 is fixed throughout. 
 *  x is scaled to [0, 1] to avoid Vandermonde ill-conditioning with float.
 *  Repeats N_REPEATS times per degree per budget.
 *  Also records the theoretically required k for each degree at
 *      epsilon = 0.5, p = 0.01, for comparison against the fixed budget.
 *
 * Output:
 *  results/exp3.csv
 *  columns: index, n, epsilon, d, m, k_budget, k_theory, t,
 *    repeat, time_mu_s, model_error
 *
 * Usage:
 *  make exp3
 * ========================================================================== */

#include "experiments.h"

#define EXP3_CSV  "results/exp3.csv"
#define DEGREE_MIN 2
#define DEGREE_MAX 10
#define DEGREE_STEPS (DEGREE_MAX - DEGREE_MIN + 1)
#define EPS_FIXED  0.5f
#define N_BUDGETS  2

static const int K_BUDGETS[N_BUDGETS] = {100, 1000};

int main(void) {
    srand((unsigned int) time(NULL));

    /* open output file */
    FILE* fp = fopen(EXP3_CSV, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s\n", EXP3_CSV);
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

           printf("degree=%d n_params=%d k_theory=%d\n",
             deg, n_params, k_theory);

           for (int r = 0; r < N_REPEATS; r++) {
            float points_x[N_TOTAL], points_y[N_TOTAL];
            float t;

            /* generate noisy data with outliers, no structural bias */
            make_data(points_x, points_y,
               n_inliers, n_outliers,
               true_params, n_params,
               NOISE_STD, &t,
               0, NULL, 0.0f);

            /* run ransac with fixed k budget */
            RansacResult res = run_ransac(points_x, points_y,
                    N_TOTAL, n_params,
                    true_params,
                    EPS_FIXED, t, d, k_budget,
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
    printf("\nExperiment 3 done. Results written to %s\n", EXP3_CSV);
    return 0;
}