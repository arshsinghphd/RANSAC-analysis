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


def points_to_line_distances(points_x, points_y, n_points, slope, intercept,
        distances):
    """
    This function takes two lists of size n_points, and a linear model as
    slope and intercept. It estimates the absolute value of distance of each
    point represented by points_x and points_y from the linear model using the
    geometric formula and fills the list distances in place.

    𝑑istances[i] =  abs((slope * points_x[i] - points_y[i] + intercept
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
