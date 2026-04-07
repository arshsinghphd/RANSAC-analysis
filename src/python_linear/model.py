"""
Model fitting functions for RANSAC line fitting pipeline. Provides a stitching
function for combining inliers from two overlapping graphs into a single
refined model analogous to what will be done in homograhy.
Also provides two suport functions for use in the RANSAC algorithm:
    least squares line fitting and
    perpendicular distance computation.
Also provides a function to estimate the euclidean distance of estimated model
from the known true model, if known.

Functions:
    stitch_models               combines inliers from two graphs into a
                                single refined linear model
    fit_line                    fits a line to n points using least squares
    points_to_line_distances    computes perpendicular distances from points
                                to a line
    model_error                 measures Euclidean distance between estimated
                                and true model parameters

"""

import math

def find_inliers(points_x, points_y, n_points, slope, intercept, threshold,
                 inliers_x, inliers_y):
    """
    Suport function for stitch_graphs. Collects inliers from points_x and
    points_y by computing perpendicular distance from each point to the line
    defined by slope and intercept. Points within threshold distance are
    appended to inlier_x and inlier_y in place.

    Params:
        points_x    a list of n_points floats
        points_y    a list of n_points floats
        n_points    int, number of points to evaluate
        slope       float, slope of the model line
        intercept   float, intercept of the model line
        threshold   float, maximum perpendicular distance to qualify as inlier
        inlier_x    a list of floats, appended to in place
        inlier_y    a list of floats, appended to in place

    Returns:
        0 for success
        -1 for error if n_points < 1
        -1 for error if threshold <= 0
    """
    if n_points < 1 or threshold <= 0:
        return -1
    distances = [float("inf")] * n_points
    points_to_line_distances(points_x, points_y, n_points,
                                   slope, intercept, distances)
    for i in range(n_points):
        if distances[i] < threshold:
            inliers_x.append(points_x[i])
            inliers_y.append(points_y[i])
    return 0


def stitch_models(points_x1, points_y1, n1,
                  slope1, intercept1,
                  points_x2, points_y2, n2,
                  slope2, intercept2,
                  threshold,
                  list_slopes, list_intercepts, pos):
    """
    Combines inliers from two overlapping graphs into a single refined linear
    model. Collects inliers from each graph using their respective RANSAC-
    recovered models and threshold, then refits one line to all combined
    inliers using least squares. This implements the stitching step — if both
    graphs share the same underlying model, the combined fit is more accurate
    than either individual fit since it uses more inlier points.

    Params:
        points_x1       a list of n1 floats, x values of graph 1
        points_y1       a list of n1 floats, y values of graph 1
        n1              int, number of points in graph 1
        slope1          float, RANSAC-recovered slope for graph 1
        intercept1      float, RANSAC-recovered intercept for graph 1
        points_x2       a list of n2 floats, x values of graph 2
        points_y2       a list of n2 floats, y values of graph 2
        n2              int, number of points in graph 2
        slope2          float, RANSAC-recovered slope for graph 2
        intercept2      float, RANSAC-recovered intercept for graph 2
        threshold       float, inlier distance threshold t
        list_slopes     a list of floats of size at least pos + 1
        list_intercepts a list of floats of size at least pos + 1
        pos             int, position to store result in list_slopes
                        and list_intercepts

    Returns:
        int, total number of inliers used in the combined fit
        -1 for error if n1 < 2
        -1 for error if n2 < 2
        -1 for error if threshold <= 0
        -1 for error if pos < 0
        -1 for error if no inliers found in either graph
    """
    if n1 < 2 or n2 < 2 or threshold <= 0 or pos < 0:
        return -1
    # find inliers from graph1    
    inliers_x1 = []
    inliers_y1 = []
    find_inliers(points_x1, points_y1, n1, slope1, intercept1, threshold,
                 inliers_x1, inliers_y1)
    n_inliers1 = len(inliers_x1)
    if n_inliers1 == 0:
        return -1
    # find inliers from graph2
    inliers_x2 = []
    inliers_y2 = []
    find_inliers(points_x2, points_y2, n2, slope2, intercept2, threshold,
                 inliers_x2, inliers_y2)
    n_inliers2 = len(inliers_x2)
    if n_inliers2 == 0:
        return -1
    # find all inliers
    n_inliers =  n_inliers1 + n_inliers2
    inliers_x = [float("inf")] * (n_inliers)
    inliers_y = [float("inf")] * (n_inliers)
    for i in range(n_inliers):
        if i < n_inliers1:
            inliers_x[i] = inliers_x1[i]
            inliers_y[i] = inliers_y1[i]
        else:
            inliers_x[i] = inliers_x2[i - n_inliers1]
            inliers_y[i] = inliers_y2[i - n_inliers1]
    # fit_line on all inliers
    fit_line(inliers_x, inliers_y, n_inliers, list_slopes, list_intercepts, pos)
    return 0


def fit_line(points_x, points_y, n_points, list_slopes, list_intercepts, pos):
    """
    This functions takes two lists of size n_points and estimates two parameters
    Slope and intercept for the model:

        points_y[i] = slope * points_x[i] + intercept;

    slope = ((n * sum(points_x[i] * [points_y[i]) - sum(points_x) * sum(points_y))
            / (n * sum(points_x[i]**2) - (sum(points_x[i]))**2)

    intercept = ((sum(points_y) - slope * sum(points_x))/n)

    It does so my minimizing the squared errors and adds these to the
    list_slopes and list_intercepts, respectively, at the indices pos in place.

    Params:
        points_x    a list of n_points floats
        points_y    a list of n_points floats
        n_points    int, number of points we need to fit the line over
        list_slopes a list of floats of size at least pos
        list_intercepts
                    a list of floats of size at least pos
        pos         int

    Returns:
        None
    """
    if n_points < 2 or pos < 0 or max(points_x) == min(points_x):
        return -1
    sum_xy = sum([points_x[i] * points_y[i] for i in range(n_points)])
    sum_y = sum(points_y)
    sum_x = sum(points_x)
    sum_x_sum_y = sum_x * sum_y
    sum_x2 = sum([points_x[i] * points_x[i] for i in range(n_points)])
    sum_x_2 = sum_x * sum_x

    list_slopes[pos] = ( (n_points * sum_xy - sum_x_sum_y)/
                         (n_points * sum_x2 - sum_x_2))
    list_intercepts[pos] = (sum_y - list_slopes[pos] * sum_x) / n_points
    return 0


def points_to_line_distances(points_x, points_y, n_points, slope, intercept,
        distances):
    """
    This function takes two lists of size n_points, and a linear model as
    slope and intercept. It estimates the absolute value of distance of each
    point represented by points_x and points_y from the linear model using the
    geometric formula and fills the list distances in place.

    𝑑istances[i] =  abs((slope * points_x[i] - points_y[i] + intercept)/
                        (math.sqrt(1 + slope * slope)))

    Params:
        points_x    a list of n_points floats
        points_y    a list of n_points floats
        n_points    int
        slope       float
        intercept   float
        distances    a list of n_points floats

    Returns:
        None
    """
    if n_points <= 0:
        return -1
    for i in range(n_points):
        distances[i] = abs((slope * points_x[i] - points_y[i] + intercept)/
                        (math.sqrt(1 + slope * slope)))
    return 0


def model_error(slope, intercept, true_slope, true_intercept):
    """
    This function measures the distance between the true model and the model
    estimated by RANSAC. It is measured as:

        math.sqrt((slope - true_slope)**2 + (intercept - true_intercept)**2)

    Params:
        slope           float, estimated slope
        intercept       float, estimated intercept
        true_slope      float, used in generating inliers
        true_intercept  float, used in generating inliers

    Returns:
        float           model error
    """
    return math.sqrt((slope - true_slope)**2 + (intercept - true_intercept)**2)
