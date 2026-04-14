/* =============================================================================
 * experiment0_2.c
 *
 * Experiment 0_2: Does wall-clock time grow with epsilon for fixed k?
 *
 *     Fixes k at the value computed for epsilon = 0.5. Varies epsilon
 *     from 0.05 to 0.95 in steps of 0.05. Runs for linear (m = 2) and
 *     quadratic (m = 3) models. N_TOTAL = 1000 is fixed throughout.
 *     Repeats N_REPEATS times per condition.
 *
 * Output:
 *     results/exp02.csv
 *     columns: index, n, epsilon, d, m, k, t, repeat, time_mu_s, model_error
 *
 * Usage:
 *     make exp02
 * ========================================================================== */

#include "experiments.h"

#define EXP02_CSV "results/exp02.csv"
// number of epsilon values
#define EPSILON_STEPS 19
// two models: linear (m=2) and quadratic (m=3)
#define N_MODELS 2
#define EPS_FOR_K 0.5f

// epsilon from 0.05 to 0.95 in steps of 0.05
static const float EPSILONS[EPSILON_STEPS] = {
    0.05f, 0.1f, 0.15f, 0.2f, 0.25f, 0.3f, 0.35f, 0.4f, 0.45f, 0.5f,
    0.55f, 0.6f, 0.65f, 0.7f, 0.75f, 0.8f, 0.85f, 0.9f, 0.95f
};


int main(void) {
    srand((unsigned int) time(NULL));

    // open csv for writing — created fresh each run
    FILE* fp = fopen(EXP02_CSV, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s\n", EXP02_CSV);
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

    int index = 0;

    for (int m = 0; m < N_MODELS; m++) {
        int n_params = n_params_list[m];
        const float* true_params = true_params_list[m];

        // fix k once per model using EPS_FOR_K=0.5 — k does not change as epsilon varies
        int k = compute_k(EPS_FOR_K, n_params, FAIL_PROB);
        printf("\n--- m=%d  k=%d ---\n", n_params, k);

        // very epsilon — this changes n_inliers, n_outliers, and d each step
        for (int e = 0; e < EPSILON_STEPS; e++) {
            float epsilon = EPSILONS[e];
            // more outliers as epsilon grows
            int n_inliers = (int)((1.0f - epsilon) * N_TOTAL);
            int n_outliers = N_TOTAL - n_inliers;
            // d shrinks as epsilon grows — fewer expected inliers
            int d = compute_d(epsilon, N_TOTAL);

            // repeat each (m, epsilon) condition for stable time estimates
            for (int r = 0; r < N_REPEATS; r++) {
                float points_x[N_TOTAL], points_y[N_TOTAL];
                // t computed from data by make_data — varies with epsilon
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
                                             epsilon, t, d, k,
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
    printf("\nExperiment 02 done. Results written to %s\n", EXP02_CSV);
    return 0;
}