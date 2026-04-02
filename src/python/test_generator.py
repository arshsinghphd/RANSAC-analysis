"""
These are tests for generator methods. There are five methods to be tested:
    make_inliers
    add_gaussian_noise
    add_laplace_noise
    add_structural_bias
    add_outliers
    
Run from terminal as follows:
    $ python -m unittest test_generator.py
"""
import generator

import unittest

class TestMakeInliers(unittest.TestCase):
    """
    For make_inliers, I am testing the following cases. Assume, n_points = n + 1.

    Happy paths:
        slope = -1, intercept = n should create two identical list of [n, ..., 0]
            should create two identical list of n points [0, .. n],
            corresponding to points (0, 0), ... , (n, n)
        Tests:
            works for negative slope.
            should creates n_points points
            all x should be within x_min and x_max
            all y should be within y_min and y_max

         slope = 0.5, intercept = 0, x_min = 0, x_max = n, y_min = 0, y_max = n/2
            should create data_x as [0, 1, ... , n], and data_y as [0, 0.5, ..., n/2]
            corresponding to points (0, 0), (1, 0.5), ... , (n, n/2)
        Tests:
            works well for fractions.
            should creates n_points points
            all x should be within x_min and x_max
            all y should be within y_min and y_max        

    Special cases:
        slope =  0, intercept = 0 should create two lists of [0, ... , 0] (n times)

    Edges:
        n = 0 should return an empty list
        n = 1 should return an empty list
        if x_min = x_max,
            return empty list
    """
    
    def test_unit_slope_zero_intercept(self):
        """
        slope =  1, intercept = 0, x_min = 0, x_max = n, y_min = 0, y_max = n
            should create two identical list of n points [0, .. n],
            corresponding to points (0, 0), ... , (n, n)
        Tests:
            should creates n_points points
            all x should be within x_min and x_max
            all y should be within y_min and y_max
        """
        # initiate all arguments
        n_inliers = 10
        data_x = [float("INF")] * n_inliers
        data_y = [float("INF")] * n_inliers
        slope = 1
        intercept = 0
        x_min = 0
        x_max = n_inliers - 1

        # fill data_x, data_y, passing them to make_inliers
        generator.make_inliers(data_x, data_y, n_inliers, slope, intercept,
                              x_min, x_max)

        # expected values
        exp_x = [float(i) for i in range(n_inliers)]
        exp_y = [float(i) for i in range(n_inliers)]      

        # assertions
        for i in range(n_inliers):
            self.assertAlmostEqual(data_x[i], exp_x[i])
            self.assertAlmostEqual(data_y[i], exp_y[i])


if __name__ == '__main__':
    unittest.main()



