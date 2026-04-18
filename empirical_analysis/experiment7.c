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
