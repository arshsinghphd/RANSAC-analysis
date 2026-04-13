/* =============================================================================
 *
 * Empirical analysis for RANSAC — Experiment 1.
 *
 * Experiment 1: How does RANSAC break down as outlier fraction increases?
 *
 *     Fix k at the value computed for epsilon = 0.5.
 * 	   Varies epsilon from 
 * 			0.1 to 0.9 in steps of 0.1.
 * 	   Runs for 
 * 			1. linear (n_params = 2), and
 *          2. quadratic (n_params = 3) models. 
 *     Repeats N_REPEATS times per condition. 
 *     Records model error and wall-clock time per run.
 *
 * Output:
 *     results/exp1.csv
 *
 * CSV columns:
 *     index, N, epsilon, d, m (n_params), k, t (threshold), repeat, time, and
 *			model_error			
 *
 * Usage:
 *     make run
 *
 * References:
 *     Fischler, M. A. and Bolles, R. C. 1981. Random sample consensus.
 *     Commun. ACM 24, 6, 381-395.
 * ========================================================================== */