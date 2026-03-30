def ransac(points_x, points_y, n_points, n_params, k_resample, threshold, 
    expected_inliers, return_array):
    """
    This function finds a random sample consensus model for the given set of 
    n_points number of points as two lists points_x and points_y. 
    The model needs to estimate n_params.
    We will randomly resample k_resample number of times to choose the best model,
    unless some model has expected_inliers number of inliers based on threshold
    distance from the estimated model which is returned immediately as a named 
    tuple. For example for a line the named tuple will have slope and intercept.

    Return array is modified in place. 

    Params:
        points_x    a list of n_points floats
        points_y    a list of n_points floats
        n_points    int
        n_params    the number of parameters to be estimated
        k_resample  chosen number of random reestimations of model
        threshold   the distance from model that is acceptable
        expected_inliers
                    int, the number of inliers expected in the data an way to 
                    early stop the search for best model.
        return_array    # parallel to a struct in C
                    n_points
                    n_params
                    k_resamples
                    threshold,
                    expected_inliers,
                    next n_params entries: n_params in some logical order, for
                    linear model:
                        slope
                        intercept
                    number of inliers
                    number of iterations actually run
    
    Returns:
        None        
    """
    pass