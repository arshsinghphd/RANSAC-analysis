import math

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
