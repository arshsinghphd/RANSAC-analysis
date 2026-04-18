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

#define EXP7_CSV "results/exp7.csv"
#define EPS_FIXED 0.3f
#define N_PARAMS 2
#define N_MIN 2
#define N_MAX 8192   // 2 ** 13
#define T_FIXED 1.0f

int main(void) {
    srand((unsigned int) time(NULL));

    FILE* fp = fopen(EXP7_CSV, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s\n", EXP7_CSV);
        return 1;
    }

    /* CSV header */
    fprintf(fp, "index,n,epsilon,d,m,k,t,repeat,time_mu_s,model_error\n");

    float true_params[] = _TRUE_PARAMS_LINEAR;
    // 10 times to give a wide berth
    int k = 10 * compute_k(EPS_FIXED, N_PARAMS, FAIL_PROB);
    int index = 0;

    for (int n_total = N_MIN; n_total <= N_MAX; n_total *= 2) {
        int n_inliers  = (int)((1.0f - EPS_FIXED) * n_total);
        int n_outliers = n_total - n_inliers;

        /* guard: need at least n_params inliers to fit */
        if (n_inliers < N_PARAMS) {
            printf("skipping n=%d: n_inliers=%d < n_params=%d\n",
                   n_total, n_inliers, N_PARAMS);
            continue;
        }

        int d = compute_d(EPS_FIXED, n_total);

        /* allocate point arrays on heap to avoid stack overflow */
        float* points_x = (float*) malloc(n_total * sizeof(float));
        float* points_y = (float*) malloc(n_total * sizeof(float));
        
        if (!points_x || !points_y) {
            fprintf(stderr, "Error: malloc failed at n=%d\n", n_total);
            free(points_x);
            free(points_y);
            break;
        }

        printf("n=%d  n_inliers=%d  n_outliers=%d  d=%d\n",
               n_total, n_inliers, n_outliers, d);

        for (int r = 0; r < N_REPEATS; r++) {
            float t;

            /* generate noisy data with outliers, no structural bias */
            make_data(points_x, points_y,
                      n_inliers, n_outliers,
                      true_params, N_PARAMS,
                      NOISE_STD, &t,
                      0, NULL, 0.0f);

            /* run ransac and collect result, use T_FIXED */
            RansacResult res = run_ransac(points_x, points_y,
                                          n_total, N_PARAMS,
                                          true_params,
                                          EPS_FIXED, T_FIXED, d, k,
                                          r, index++);

            /* write one CSV row */
            fprintf(fp, "%d,%d,%.1f,%d,%d,%d,%.6f,%d,%.2f,%.6f\n",
                    res.index, res.n, res.epsilon,
                    res.d, res.m, res.k, res.t,
                    res.repeat, res.time_mu_s, res.model_error);
        }
        free(points_x);
        free(points_y);
    }

    fclose(fp);
    printf("Experiment 7 done. Results written to %s\n", EXP7_CSV);
    return 0;
}
