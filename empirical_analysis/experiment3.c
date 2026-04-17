/* =============================================================================
 * experiments3.c
 *
 * Experiment 3: Does wall-clock time vary with inlier threshold t?
 *
 *     Fixes k at the value computed for epsilon = 0.5 and fixes
 *     epsilon = 0.5. Varies t as a multiplier of noise_std from 0.5
 *     to 5.0 in steps of 0.5. Runs for linear (m = 2) and quadratic
 *     (m = 3) models. N_TOTAL = 1000 is fixed throughout.
 *     Repeats N_REPEATS times per condition.
 *
 * Output:
 *     results/exp3.csv
 *     columns: index, n, epsilon, d, m, k, t, t_multiplier,
 *              repeat, time_mu_s, model_error
 *
 * Usage:
 *     make exp3
 * ========================================================================== */

#include "experiments.h"

#define EXP3_CSV "results/exp3.csv"
#define T_STEPS 10
#define N_MODELS 2
#define EPS_FIXED 0.5f

static const float T_MULTIPLIERS[T_STEPS] = {
    0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f, 5.0f
};


int main(void) {
    srand((unsigned int) time(NULL));

    FILE* fp = fopen(EXP3_CSV, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s\n", EXP3_CSV);
        return 1;
    }

    fprintf(fp, "index,n,epsilon,d,m,k,t,t_multiplier,"
                "repeat,time_mu_s,model_error\n");

    int n_params_list[N_MODELS] = {2, 3};
    float true_params_linear[] = _TRUE_PARAMS_LINEAR;
    float true_params_quadratic[] = _TRUE_PARAMS_QUADRATIC;
    const float* true_params_list[N_MODELS] = {
        true_params_linear,
        true_params_quadratic
    };

    int n_inliers = (int)((1.0f - EPS_FIXED) * N_TOTAL);
    int n_outliers = N_TOTAL - n_inliers;
    int d = compute_d(EPS_FIXED, N_TOTAL);
    int index = 0;

    for (int m = 0; m < N_MODELS; m++) {
        int n_params = n_params_list[m];
        const float* true_params = true_params_list[m];

        int k = compute_k(EPS_FIXED, n_params, FAIL_PROB);
        printf("\n--- m=%d  k=%d ---\n", n_params, k);

        for (int ti = 0; ti < T_STEPS; ti++) {
            float t_mult = T_MULTIPLIERS[ti];
            float t = t_mult * NOISE_STD;

            for (int r = 0; r < N_REPEATS; r++) {
                float points_x[N_TOTAL], points_y[N_TOTAL];
                float t_unused;

                make_data(points_x, points_y,
                            n_inliers, n_outliers,
                            true_params, n_params,
                            NOISE_STD, &t_unused,
                            0, NULL, 0.0f);

                RansacResult res = run_ransac(points_x, points_y,
                                                N_TOTAL, n_params,
                                                true_params,
                                                EPS_FIXED, t, d, k,
                                                r, index++);

                fprintf(fp, "%d,%d,%.2f,%d,%d,%d,%.4f,%.1f,%d,%.2f,%.6f\n",
                        res.index, res.n, res.epsilon,
                        res.d, res.m, k, t, t_mult,
                        res.repeat, res.time_mu_s, res.model_error);
            }
        }
    }
    fclose(fp);
    printf("\nExperiment 3 done. Results written to %s\n", EXP3_CSV);
    return 0;
}