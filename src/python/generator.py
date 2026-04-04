import math
import numpy
import random


def make_inliers(data_x, data_y, n_inliers, slope, intercept, x_min, x_max):
    """
    Fills the lists inliers_x, inliers_y in place based on input.

    Params:
        data_x      a list of floats of size >= n_points
                    # size od data_x, data_y = n_inliers + n_outliers
        data_y      a list of floats of size >= n_points
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
        data_x[i] = x_min + i * step
        data_y[i] = slope * data_x[i] + intercept
    return 0        # success


def box_muller(sigma):
    """
    This function returns a random number drawn from a bell shaped distribution
    with mean 0 and standard deviation of sigma. It employs Box-Muller
    transform.

    Background: The Box-Muller transform is a mathematical method for generating pairs of
    independent, standard normally distributed (Gaussian) random numbers from
    two uniformly distributed random numbers between 0 and 1. It transforms
    uniform samples using logarithmic and trigonometric functions to map them
    onto a bell curve, offering a computationally efficient alternative to
    inverse transform sampling.
    Source: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform

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


def add_gaussian_noise(data_y, n_inliers, std):
    """
    Adds zero mean gaussian noise to data_y inplace based on user's inputs.

    Params:
        data_y      a list of floats
        n_inliers   int, number of inliers to be modified
        std         float, standard deviation of gaussian noise

    Returns:
        0 for success
        -1 for error
    """
    if n_inliers < 2:
        return -1
    for i in range(n_inliers):
        data_y[i] += box_muller(std)
    return 0


def add_laplace_noise(data_y, n_inliers, scale_noise):
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
        data_y      a list of floats
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
        data_y[i] += z
    return 0


def add_outliers(data_x, data_y, n_inliers, n_outliers, x_min, x_max,
                 y_min, y_max):
    """
    Adds outliers to the data_x and data_y inplace.
    Outliers are classification or gross error.
    This adds n_outlier points drawn uniformly from the space of points.

    Params:
        data_x      a list of floats
        data_y      a list of floats
        n_inliers   size of existing valid points in data
        n_outliers  int, no. of outliers to add starting at index n_inliers
        x_min       float, lower limit of x
        x_max       float, upper limit of x
        y_min       float, lower limit of y
        y_max       float, upper limit of y

    Returns:
        0 for success
        -1 for error
    """
    if (n_inliers < 0 or n_outliers < 0 or n_inliers + n_outliers < 2 or
        x_min == x_max or y_min == y_max):
        return -1
    for i in range(n_outliers):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        data_x.append(x)
        data_y.append(y)
    return 0


def add_structural_bias(data_y, data_x, n_inliers, bias_fn):
    """
    Adds structural bias to the data inplace based on user's inputs.

    Params:
        data_y      a list of floats
        data_x      a list of floats
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
        data_y[i] += bias_fn(data_x[i])
    return 0

