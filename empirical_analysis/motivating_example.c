/* =============================================================================
 * Motivating Example
 * I create two overlapping graphs with the same true linear model.
 * I recover the underlying models in two ways:
    1. using fit_model over the appended dataset which is like solving
       naive OLS, and
    2. using stitch_models which is the RANSAC approach
 * Later I will plot this data and the models recovered using naive OLS and
 * the RANSAC parameters to visually compare the two approaches and
 * demonstrate the goodness of fit.
 * ========================================================================== */

#include "experiments.h"

#define N 100
#define N_PARAMS 2
// graph 1 x range
#define X_MIN1 0.0f
#define X_MAX1 0.6f
// graph 2 x range
#define X_MIN2 0.39f
#define X_MAX2 0.99f
// fraction of outliers
#define EPS 0.1f
#define EXAMPLE_CSV "results/example.csv"


int main(void) {
    srand((unsigned int) time(NULL));

    // true params same for both graphs
    float true_params[] = _TRUE_PARAMS_LINEAR;

    // n_inliers and n_outliers same for both
    int n_inliers  = (int)((1.0f - EPS) * N);
    int n_outliers = N - n_inliers;

    // create graph 1: x in [X_MIN1, X_MAX1]
    float points_x1[N], points_y1[N];
    // generate inliers on true model
    make_inliers(points_x1, points_y1, n_inliers,
                 true_params, N_PARAMS, X_MIN1, X_MAX1);
    // add gaussian noise to inliers
    add_gaussian_noise(points_y1, n_inliers, NOISE_STD);
    // estimate threshold from inlier data before adding outliers
    float t1 = compute_t(points_x1, points_y1, n_inliers, N_PARAMS);
    if (t1 < 1e-6f) t1 = 2.0f * NOISE_STD;
    // append outliers
    add_outliers(points_x1, points_y1, n_inliers, n_outliers,
                 true_params, N_PARAMS, NOISE_STD);

    // create graph 2: x in [X_MIN2, X_MAX2]
    float points_x2[N], points_y2[N];
    // generate inliers on true model
    make_inliers(points_x2, points_y2, n_inliers,
                 true_params, N_PARAMS, X_MIN2, X_MAX2);
    // add gaussian noise to inliers
    add_gaussian_noise(points_y2, n_inliers, NOISE_STD);
    // estimate threshold from inlier data before adding outliers
    float t2 = compute_t(points_x2, points_y2, n_inliers, N_PARAMS);
    if (t2 < 1e-6f) t2 = 2.0f * NOISE_STD;
    // append outliers
    add_outliers(points_x2, points_y2, n_inliers, n_outliers,
                 true_params, N_PARAMS, NOISE_STD);

    // OLS on each graph separately
    float ols_params1[N_PARAMS], ols_params2[N_PARAMS];
    fit_model(points_x1, points_y1, N, ols_params1, N_PARAMS);
    fit_model(points_x2, points_y2, N, ols_params2, N_PARAMS);

    // RANSAC on each graph separately
    int k  = compute_k(EPS, N_PARAMS, FAIL_PROB);
    int d  = compute_d(EPS, N);
    float ransac_return1[2 + N_PARAMS];
    float ransac_return2[2 + N_PARAMS];
    ransac(points_x1, points_y1, N, N_PARAMS, k, t1, d, ransac_return1);
    ransac(points_x2, points_y2, N, N_PARAMS, k, t2, d, ransac_return2);

    // extract ransac params from return array
    float ransac_params1[N_PARAMS], ransac_params2[N_PARAMS];
    for (int j = 0; j < N_PARAMS; j++) {
        ransac_params1[j] = ransac_return1[2 + j];
        ransac_params2[j] = ransac_return2[2 + j];
    }

    // OLS on combined data
    float x_all[N + N], y_all[N + N];
    for (int i = 0; i < N; i++) {
        x_all[i] = points_x1[i];
        y_all[i] = points_y1[i];
    }
    for (int i = 0; i < N; i++) {
        x_all[N + i] = points_x2[i];
        y_all[N + i] = points_y2[i];
    }
    float ols_combined[N_PARAMS];
    fit_model(x_all, y_all, N + N, ols_combined, N_PARAMS);

    // stitch RANSAC models
    float stitched_params[N_PARAMS];
    float threshold = (t1 + t2) / 2.0f;
    stitch_models(points_x1, points_y1, N, ransac_params1,
                  points_x2, points_y2, N, ransac_params2,
                  stitched_params, N_PARAMS, threshold);

    // print results
    printf("True params:  a0=%.4f  a1=%.4f\n",
           true_params[0], true_params[1]);
    printf("\nOLS graph 1:  a0=%.4f  a1=%.4f  error=%.4f\n",
           ols_params1[0], ols_params1[1],
           model_error(ols_params1, true_params, N_PARAMS));
    printf("OLS graph 2:  a0=%.4f  a1=%.4f  error=%.4f\n",
           ols_params2[0], ols_params2[1],
           model_error(ols_params2, true_params, N_PARAMS));
    printf("OLS combined:  a0=%.4f  a1=%.4f  error=%.4f\n",
           ols_combined[0], ols_combined[1],
           model_error(ols_combined, true_params, N_PARAMS));
    printf("\nRANSAC graph 1: a0=%.4f  a1=%.4f  error=%.4f\n",
           ransac_params1[0], ransac_params1[1],
           model_error(ransac_params1, true_params, N_PARAMS));
    printf("RANSAC graph 2:  a0=%.4f  a1=%.4f  error=%.4f\n",
           ransac_params2[0], ransac_params2[1],
           model_error(ransac_params2, true_params, N_PARAMS));
    printf("RANSAC stitched:  a0=%.4f  a1=%.4f  error=%.4f\n",
           stitched_params[0], stitched_params[1],
           model_error(stitched_params, true_params, N_PARAMS));

    // write CSV
    FILE* fp = fopen(EXAMPLE_CSV, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s\n", EXAMPLE_CSV);
        return 1;
    }
    fprintf(fp, "x,y,graph\n");
    for (int i = 0; i < N; i++)
        fprintf(fp, "%.4f,%.4f,1\n", points_x1[i], points_y1[i]);
    for (int i = 0; i < N; i++)
        fprintf(fp, "%.4f,%.4f,2\n", points_x2[i], points_y2[i]);
    fclose(fp);

    printf("\nData written to %s\n", EXAMPLE_CSV);
    return 0;
}