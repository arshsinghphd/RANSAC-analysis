def count_inliers(distances, n_points, threshold):
    """
    This function returns the number of distances that are within threshold
    distance.

    Params:
        distances   a list of floats of size n_points
        n_points    int, number of points in distances
        threshold   the threshold distance below which the point is considered 
                    an inlier        

    Returns:
        int         num of inliers
    """
    pass


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
    pass
