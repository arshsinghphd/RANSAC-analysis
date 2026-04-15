/* =============================================================================
 * experiments5.c
 *
 * Experiment 5: At what structural bias probability does RANSAC fail?
 *
 *      Fixes epsilon = 0.3 and varies pr from 0.0 to 1.0 in steps of 0.05.
 *      Runs for three bias types: constant, linear, periodic.
 *      Runs for linear (m = 2) and quadratic (m = 3) models.
 *      N_TOTAL = 1000 is fixed throughout.
 *      Repeats N_REPEATS times per condition.
 *
 * Output:
 *      results/exp5.csv
 *      columns: index, n, epsilon, d, m, k, t, pr, bias_type,
 *          repeat, time_mu_s, model_error
 *
 * Usage:
 *      make exp5
 * ========================================================================== */

#include "experiments.h"

#define EXP5_CSV  "results/exp5.csv"
#define PR_STEPS  21
#define N_MODELS  2
#define N_BIAS_TYPES 3
#define EPS_FIXED  0.1f 

static const float PR_VALUES[PR_STEPS] = {
    0.00f, 0.05f, 0.10f, 0.15f, 0.20f, 0.25f, 0.30f, 0.35f, 0.40f,
    0.45f, 0.50f, 0.55f, 0.60f, 0.65f, 0.70f, 0.75f, 0.80f, 0.85f,
    0.90f, 0.95f, 1.00f
};

/* bias function table */
typedef struct {
    const char* name;
    float  (*fn)(float);
} BiasEntry;

static const BiasEntry BIAS_TYPES[N_BIAS_TYPES] = {
    {"constant", bias_constant },
    {"linear", bias_linear_bias},
    {"periodic", bias_periodic },
};


int main(void) {
    srand((unsigned int) time(NULL));

    /* open output file */
    FILE* fp = fopen(EXP5_CSV, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s\n", EXP5_CSV);
        return 1;
    }

    /* CSV header */
    fprintf(fp, "index,n,epsilon,d,m,k,t,pr,bias_type,"
    "repeat,time_mu_s,model_error\n");

    /* model configurations */
    int n_params_list[N_MODELS] = {2, 3};
    float true_params_linear[] = _TRUE_PARAMS_LINEAR;
    float true_params_quadratic[] = _TRUE_PARAMS_QUADRATIC;
    const float* true_params_list[N_MODELS] = {
        true_params_linear,
        true_params_quadratic
    };

    /* fixed epsilon and derived parameters */
    int n_inliers = (int)((1.0f - EPS_FIXED) * N_TOTAL);
    int n_outliers = N_TOTAL - n_inliers;
    int d = compute_d(EPS_FIXED, N_TOTAL);

    int index = 0;

    for (int m = 0; m < N_MODELS; m++) {
    int n_params = n_params_list[m];
    const float* true_params = true_params_list[m]; 
    /* fix k at epsilon = 0.5, matching exp1 */
    int k = compute_k(0.5f, n_params, FAIL_PROB);

    printf("model m=%d k=%d\n", n_params, k);

    for (int b = 0; b < N_BIAS_TYPES; b++) {
        const char* bias_name = BIAS_TYPES[b].name;
        float (*bias_fn)(float) = BIAS_TYPES[b].fn;

        for (int p = 0; p < PR_STEPS; p++) {
            float pr = PR_VALUES[p];

            for (int r = 0; r < N_REPEATS; r++) {
                float points_x[N_TOTAL], points_y[N_TOTAL];
        
                float t;

                /* generate noisy data with outliers and structural bias */
                make_data(points_x, points_y, n_inliers, n_outliers,
                        true_params, n_params,
                        NOISE_STD, &t, 1, bias_fn, pr);

                /* run ransac and collect result */
                RansacResult res = run_ransac(points_x, points_y,
                     N_TOTAL, n_params,
                     true_params,
                     EPS_FIXED, t, d, k,
                     r, index++);

                /* write one CSV row */
                fprintf(fp, "%d,%d,%.2f,%d,%d,%d,%.4f,%.2f,%s,"
                "%d,%.2f,%.6f\n",
                res.index, res.n, res.epsilon,
                res.d, res.m, res.k, res.t, pr, bias_name,
                res.repeat, res.time_mu_s, res.model_error);
            }
        }
    }
 }

 fclose(fp);
 printf("Experiment 5 done. Results written to %s\n", EXP5_CSV);
 return 0;
}