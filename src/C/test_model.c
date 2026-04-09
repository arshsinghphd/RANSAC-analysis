/**
 * Tests for model.c functions:
 * 	eval_model
 * 	fit_model
 * 	find_model_inliers
 * 	stitch_models
 * 	points_to_line_distances
 * 	model_error
 * 
 * fit_model fits a polynomial model of degree n_params - 1 to n_points data
 * points using least squares via Gaussian elimination on the normal equations.
 * Coefficients are stored in params at offset pos * n_params, from lowest to
 * highest degree.
 * 
 * points_to_line_distances computes vertical distance from each point to
 * the model, storing results in distances.
 */

#include "model.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

