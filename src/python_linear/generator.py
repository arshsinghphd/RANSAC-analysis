"""
Data generation functions for the RANSAC pipeline. Provides inlier generation
on a linear model, gaussian noise injection, outlier generation guaranteed to
lie outside the inlier band, and structural bias injection.

The inlier band is defined as points within 2 * noise_std of the true line,
using perpendicular distance. Outliers are guaranteed to lie outside this band.

Functions:
    make_inliers            fills points_x, points_y with inlier data
                            evenly spaced from x_min to x_max on y = a1*x + a0
    box_muller              generates a gaussian random number via Box-Muller
    add_gaussian_noise      adds zero mean gaussian noise to points_y in place
    add_outliers            appends outlier points guaranteed outside inlier band
    add_structural_bias     adds a user-defined structural bias to points_y

References:
    Box, G. E. P. and Muller, M. E. 1958. A note on the generation of random
    normal deviates. The Annals of Mathematical Statistics 29, 2, 610-611.
"""

import math
import numpy
import random


def make_inliers(points_x, points_y, n_inliers, slope, intercept, x_min, x_max):
    """
    Fills the lists inliers_x, inliers_y in place based on input.

    Params:
        points_x    a list of floats of size >= n_points
                    # size of points_x, points_y = n_inliers + n_outliers
        points_y    a list of floats of size >= n_points
        n_inliers   int, number of inlier points to be added
        slope       float, slope of the line
        intercept   float, intercept of the line
        x_min       float, lower limit of x
        x_max       float, upper limit of x

    Returns:
        0 for success
        -1 for error
    """
    if n_inliers < 2 or x_min == x_max:
        return -1   # error code

    # infer the increment of x
    step = (x_max - x_min) / (n_inliers - 1)
    for i in range(n_inliers):
        points_x[i] = x_min + i * step
        points_y[i] = slope * points_x[i] + intercept
    return 0        # success


def box_muller(sigma):
    """
    This function returns a random number drawn from a bell shaped distribution
    with mean 0 and standard deviation of sigma. It employs Box-Muller
    transform.

    Background: The Box-Muller transform is a mathematical method for generating
    pairs of independent, standard normally distributed (Gaussian) random
    numbers from two uniformly distributed random numbers between 0 and 1. It
    transforms uniform samples using logarithmic and trigonometric functions to
    map them onto a bell curve, offering a computationally efficient
    alternative to inverse transform sampling.

    Source:
        Box, G. E. P. and Muller, M. E. 1958. A note on the generation of random
        normal deviates. The Annals of Mathematical Statistics 29, 2, 610–611.

    Params:
        sigma, float

    Returns:
        float, a random number from a mean 0 distribution with standard
        deviation of sigma.
    """
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return z * sigma


def add_gaussian_noise(points_y, n_inliers, std):
    """
    Adds zero mean gaussian noise to points_y inplace based on user's inputs.

    Params:
        points_y    a list of floats
        n_inliers   int, number of inliers to be modified
        std         float, standard deviation of gaussian noise

    Returns:
        0 for success
        -1 for error
    """
    if n_inliers < 2 or std <= 0:
        return -1
    for i in range(n_inliers):
        points_y[i] += box_muller(std)
    return 0


def add_laplace_noise(points_y, n_inliers, scale_noise):
    """
    Adds zero-mean laplacean noise to the data inplace based on user's inputs.

    Background:

        Gaussian distribution:  tails decay as exp(-x²)
        Laplace distribution:   tails decay as exp(-|x|), slower decay
        Laplace distribution has a higher probability of generating points far
        from the mean than Gaussian with the same scale. This makes it a good
        model for measurement errors that occasionally produce large deviations
        — more realistic than pure Gaussian.


    Params:
        points_y      a list of floats
        n_inliers   int, number of inliers to be modified
        scale_noise float, spread, decides how fat the tail will be

    Returns:
        0 for success
        -1 for error
    """
    if n_inliers < 2 or scale_noise <= 0:
        return -1
    for i in range(n_inliers):
        u = random.uniform(-0.5, 0.5)
        # map u on to CDF of Laplace
        z = -scale_noise * (-1 if u < 0 else 1) * math.log(1 - 2 * abs(u))
        points_y[i] += z
    return 0


def add_outliers(points_x, points_y, n_inliers, n_outliers,
                 slope, intercept, noise_std):
    """
    Adds n_outliers points to points_x and points_y in place, guaranteeing that
    every added point is a true classification error — that is, it lies
    outside the inlier band defined by the true model. The inlier band is
    defined as points within 2 * noise_std of the true line. Outliers are
    placed in a region guaranteed to be far from the inlier band by drawing
    from a range that is twice the data range above or below the data,
    chosen randomly. This guarantees that all n_outliers points are actual
    outliers and not accidental inliers.

    Params:
        points_x    a list of floats of size n_inliers, modified in place
        points_y    a list of floats of size n_inliers, modified in place
        n_inliers   int, size of existing valid points in data
        n_outliers  int, number of outliers to add
        slope       float, slope of the true model
        intercept   float, intercept of the true model
        noise_std   float, standard deviation of inlier noise, used to
                    define the inlier band as 2 * noise_std from the line

    Returns:
        0 for success
        -1 for error if n_inliers < 0
        -1 for error if n_outliers < 0
        -1 for error if n_inliers + n_outliers < 2
        -1 for error if noise_std <= 0
    """
    if (n_inliers < 0 or n_outliers < 0 or
            n_inliers + n_outliers < 2 or noise_std <= 0):
        return -1

    x_min_data = min(points_x[:n_inliers])
    x_max_data = max(points_x[:n_inliers])
    x_range = x_max_data - x_min_data if x_max_data != x_min_data else 1.0
    x_offset = 2 * x_range

    inlier_band = 2 * noise_std * math.sqrt(1 + slope * slope)

    for i in range(n_outliers):
        if random.random() < 0.5:
            x = random.uniform(x_max_data + x_range, x_max_data + x_offset)
        else:
            x = random.uniform(x_min_data - x_offset, x_min_data - x_range)

        y_on_line = slope * x + intercept
        if random.random() < 0.5:
            y = random.uniform(y_on_line + inlier_band,
                               y_on_line + inlier_band + x_offset)
        else:
            y = random.uniform(y_on_line - inlier_band - x_offset,
                               y_on_line - inlier_band)

        points_x.append(x)
        points_y.append(y)

    return 0


def add_structural_bias(points_y, points_x, n_inliers, bias_fn):
    """
    Adds structural bias to the data inplace based on user's inputs.

    Params:
        points_y    a list of floats
        points_x    a list of floats
        n_inliers   int, number of inliers to be modified
        bias_fn     a function or lambda that defines structural bias
                    e.g.
                    lambda x: 0.5 * x       # linear structural bias
                    lambda x: math.sin(x)   # periodic structural bias

    Returns:
        0 for success
        -1 for error
    """
    if n_inliers < 2 or bias_fn == None:
        return -1
    for i in range(n_inliers):
        points_y[i] += bias_fn(points_x[i])
    return 0
