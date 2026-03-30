def fit_line(points_x, points_y, n_points, list_slopes, list_intercepts, pos):
    """
    This functions takes two lists of size n_points and estimates two parameters
     Slope and intercept for the model:

        points_y[i] = slope * points_x[i] + intercept 

    by minimizing the squared errors and adds these to the list_slopes and 
    list_intercepts, respectively, at the indices pos.

    Params:
        points_x    a list of n_points floats
        points_y    a list of n_points floats
        n_points    int, number of points we need to fit the line over
        list_slopes a list of floats of size (pos - 1)
        list_intercepts
                    a list of floats of size (pos - 1)
        pos         int

    Returns: 
        None
    """
    pass

def points_to_line_distances(points_x, points_y, n_points, slope, intercept, 
        distances):
    """
    This function takes two lists of size n_points, and a linear model as 
    slope and intercept. It estimates the absolute value of distance of each 
    point represented by points_x and points_y from the linear model using the 
    geometric formula and fills the list distances in place.

    𝑑istances[i] =  abs((slope * points_x[i] - points_y[i] + intercept) / 
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
    pass