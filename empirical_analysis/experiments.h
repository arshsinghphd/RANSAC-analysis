/* =============================================================================
 * experiments.h
 *
 * Shared constants, true model parameters, and helper functions for
 * the RANSAC empirical analysis experiments. Included by experiment1.c,
 * experiment2.c, and experiment3.c.
 *
 * All experiments use synthetic two-dimensional Cartesian point sets
 * generated from known polynomial models. Total dataset size N_TOTAL
 * is fixed across all experiments to isolate the effect of the variable
 * under study.
 * ========================================================================== */

#ifndef EXPERIMENTS_H
#define EXPERIMENTS_H

#include "../src/C/generator.h"
#include "../src/C/model.h"
#include "../src/C/ransac.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#endif /* EXPERIMENTS_H */