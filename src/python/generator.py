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
        None.
    """
    pass


def add_gaussian_noise(data_y, n_inliers, std):
    """ 
    Adds zero mean gaussian noise to data_y inplace based on user's inputs.
    
    Params: 
        data_y      a list of floats
        n_inliers   int, number of inliers to be modified
        std         float, standard deviation of gaussian noise

    Returns:
        None
    """
    pass


def add_laplace_noise(data_y, n_inliers, scale_noise):
    """ 
    Adds zero-mean laplacean noise to the data inplace based on user's inputs. 

    Params:
        data_y      a list of floats
        n_inliers   int, number of inliers to be modified
        scale_noise float, spread, decides how fat the tail will be

    Returns:
        None
    """
    pass


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
        None
    """
    pass


def add_outliers(data_x, data_y, n_inliers, n_outliers, x_min, x_max, y_min, y_max):
    """
    Adds outliers to the data_x and data_y inplace. 
    Outliers are classification or gross error.
    This adds n_outlier points drawn uniformly from the space to data.

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
        None
    """
    pass

