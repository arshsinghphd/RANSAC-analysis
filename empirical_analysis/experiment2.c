/* =============================================================================
 * experiments2.c
 *
 * Experiment 2: At what structural bias probability does RANSAC fail?
 *
 *     Fixes epsilon = 0.3 and varies pr from 0.0 to 1.0 in steps of 0.05.
 *     Runs for three bias types: constant, linear, periodic.
 *     Runs for linear (m = 2) and quadratic (m = 3) models.
 *     N_TOTAL = 1000 is fixed throughout.
 *     Repeats N_REPEATS times per condition.
 *
 * Output:
 *     results/exp2.csv
 *     columns: index, n, epsilon, t, d, m, k, pr, bias_type,
 *              repeat, time_mu_s, model_error
 *
 * Usage:
 *     make exp2
 * ========================================================================== */

#include "experiments.h"

#define EXP2_CSV      "results/exp2.csv"
#define PR_STEPS      21
#define N_MODELS      2
#define N_BIAS_TYPES  3
#define EPS_FIXED     0.3f

static const float PR_VALUES[PR_STEPS] = {
    0.00f, 0.05f, 0.10f, 0.15f, 0.20f, 0.25f, 0.30f, 0.35f, 0.40f,
    0.45f, 0.50f, 0.55f, 0.60f, 0.65f, 0.70f, 0.75f, 0.80f, 0.85f,
    0.90f, 0.95f, 1.00f
};

/* bias function table */
typedef struct {
    const char*   name;
    float       (*fn)(float);
} BiasEntry;

static const BiasEntry BIAS_TYPES[N_BIAS_TYPES] = {
    {"constant", bias_constant   },
    {"linear",   bias_linear_bias},
    {"periodic", bias_periodic   },
};
