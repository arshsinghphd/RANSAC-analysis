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
    Tests happy paths, combinations of
        Positive and negative slopes
        Positive and negative intercepts

    Special cases:
        slope =  0, intercept = 0 (flat line)

    Edges less than 2 points:
        n = 0 should return -1
        n = 1 should return -1
        x_min = x_max should return -1
    """


    def test_unit_slope_zero_intercept(self):
        """
        slope = 1, intercept = 0, x_min = 0, x_max = n - 1
            should create two identical list of n points [0, ..., n - 1],
            corresponding to points (0, 0), ... , (n - 1, n - 1)
        Tests:
            should creates n_inliers points
            all x should be within x_min and x_max
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


    def test_n_less_than_2(self):
        """
        if n < 2, should return -1
        """
        n_inliers = 1
        data_x = [float("INF")] * n_inliers
        data_y = [float("INF")] * n_inliers
        slope = 1
        intercept = 0
        x_min = 0
        x_max = n_inliers - 1
        ret = generator.make_inliers(data_x, data_y, n_inliers, slope, intercept,
                              x_min, x_max)
        self.assertEqual(ret, -1)


    def test_x_min_equals_x_max(self):
        """
        if x_min == x_max, should return -1
        """
        n_inliers = 10
        data_x = [float("INF")] * n_inliers
        data_y = [float("INF")] * n_inliers
        slope = 1
        intercept = 0
        x_min = 0
        x_max = 0
        ret = generator.make_inliers(data_x, data_y, n_inliers, slope, intercept,
                              x_min, x_max)
        self.assertEqual(ret, -1)


    def test_negative_unit_slope_zero_intercept(self):
        """
        slope = -1, intercept = n, x_min = 0, x_max = n - 1
            should create data_x as [0, ... , n]
            should create data_y as [n - 1, ... , 0]
            corresponding to points (0, n - 1), ... , (n - 1, 0)
        Tests:
            works for negative slope.
            should creates n_inliers points
            all x should be within x_min and x_max
        """
        # initiate all arguments
        n_inliers = 10
        data_x = [float("INF")] * n_inliers
        data_y = [float("INF")] * n_inliers
        slope = -1
        intercept = n_inliers - 1
        x_min = 0
        x_max = n_inliers - 1

        # fill data_x, data_y, passing them to make_inliers
        generator.make_inliers(data_x, data_y, n_inliers, slope, intercept,
                               x_min, x_max)

        # expected values
        exp_x = [float(i) for i in range(n_inliers)]
        exp_y = [float(n_inliers - 1 - i) for i in range(n_inliers)]

        # assertions
        for i in range(n_inliers):
            self.assertAlmostEqual(data_x[i], exp_x[i])
            self.assertAlmostEqual(data_y[i], exp_y[i])


    def test_float_slope_zero_intercept(self):
        """
        slope = 0.5, intercept = 0, x_min = 0, x_max = n - 1
            should create data_x as [0, 1, ... , n - 1]
            should create data_y as [0, 1/2, ... , (n - 1)/2]
            corresponding to points (0, 0), ... , (n - 1, (n - 1)/2)
        Tests:
            works for fractional slope
            should creates n_inliers points
            all x should be within x_min and x_max
        """
        # initiate all arguments
        n_inliers = 10
        data_x = [float("INF")] * n_inliers
        data_y = [float("INF")] * n_inliers
        slope = 0.5
        intercept = 0
        x_min = 0
        x_max = n_inliers - 1

        # fill data_x, data_y, passing them to make_inliers
        generator.make_inliers(data_x, data_y, n_inliers, slope, intercept,
                               x_min, x_max)

        # expected values
        exp_x = [float(i) for i in range(n_inliers)]
        exp_y = [i / 2 for i in range(n_inliers)]

        # assertions
        for i in range(n_inliers):
            self.assertAlmostEqual(data_x[i], exp_x[i])
            self.assertAlmostEqual(data_y[i], exp_y[i])


    def test_unit_slope_negative_intercept(self):
        """
        slope = 1, intercept = - n + 1, x_min = 0, x_max = n - 1
            should create two list of n points
                data_x = [0, ..., n - 1],
                data_y = [-n + 1, ..., 0]
            corresponding to points (0, -n + 1), ... , (n - 1, 0)
        Tests:
            works with negative intercept
            should creates n_inliers points
            all x should be within x_min and x_max
        """
        # initiate all arguments
        n_inliers = 10
        data_x = [float("INF")] * n_inliers
        data_y = [float("INF")] * n_inliers
        slope = 1
        intercept = -n_inliers + 1
        x_min = 0
        x_max = n_inliers - 1

        # fill data_x, data_y, passing them to make_inliers
        generator.make_inliers(data_x, data_y, n_inliers, slope, intercept,
                              x_min, x_max)

        # expected values
        exp_x = [float(i) for i in range(n_inliers)]
        exp_y = [float(- n_inliers + 1 + i) for i in range(n_inliers)]

        # assertions
        for i in range(n_inliers):
            self.assertAlmostEqual(data_x[i], exp_x[i])
            self.assertAlmostEqual(data_y[i], exp_y[i])


    def test_zero_slope_zero_intercept(self):
        """
        slope = 0, intercept = 0, x_min = 0, x_max = n - 1
            should create two list of n points
                data_x = [0, ..., n - 1],
                data_y = [0, ..., 0]
            corresponding to points (0, 0), ... , (0, 0)
        Tests:
            Special case
            should creates n_inliers points
            all x should be within x_min and x_max
        """
        # initiate all arguments
        n_inliers = 10
        data_x = [float("INF")] * n_inliers
        data_y = [float("INF")] * n_inliers
        slope = 0
        intercept = 0
        x_min = 0
        x_max = n_inliers - 1
        # fill data_x, data_y, passing them to make_inliers
        generator.make_inliers(data_x, data_y, n_inliers, slope, intercept,
                              x_min, x_max)
        # expected values
        exp_x = [float(i) for i in range(n_inliers)]
        exp_y = [float(0)] * n_inliers
        # assertions
        for i in range(n_inliers):
            self.assertAlmostEqual(data_x[i], exp_x[i])
            self.assertAlmostEqual(data_y[i], exp_y[i])


class testAddGaussianNoise(unittest.TestCase):
    """
    Tests two happy paths with std of int and float.
        all values should lie between data_y (original) +/- 3 * std
    Tests edge cases:
        if n_inliers < 2, should return -1
        if std <= 0, should return -1
    """

    def test_happy_path_int_std(self):
        """
        All noisy data_y[i] should be original data_y[i] +/- 5 * std.
        """
        # initiate all arguments
        n_inliers = 10
        data_x = [float("INF")] * n_inliers
        data_y = [float("INF")] * n_inliers
        slope = 0
        intercept = 0
        x_min = 0
        x_max = n_inliers - 1
        # fill data_x, data_y, passing them to make_inliers
        generator.make_inliers(data_x, data_y, n_inliers, slope, intercept,
                              x_min, x_max)
        copy_data_y = data_y.copy()
        # add gaussian noice
        std = 1
        generator.add_gaussian_noise(data_y, n_inliers, std)
        # assertions
        for i in range(n_inliers):
            self.assertLessEqual(abs(data_y[i] - copy_data_y[i]), 5 * std)


    def test_happy_path_float_std(self):
        """
        All noisy data_y[i] should be original data_y[i] +/- 5 * std.
        """
        # initiate all arguments
        n_inliers = 10
        data_x = [float("INF")] * n_inliers
        data_y = [float("INF")] * n_inliers
        slope = 0
        intercept = 0
        x_min = 0
        x_max = n_inliers - 1
        # fill data_x, data_y, passing them to make_inliers
        generator.make_inliers(data_x, data_y, n_inliers, slope, intercept,
                              x_min, x_max)
        copy_data_y = data_y.copy()
        # add gaussian noice
        std = (max(data_y) - min(data_y))/6
        generator.add_gaussian_noise(data_y, n_inliers, std)
        # assertions
        for i in range(n_inliers):
            self.assertLessEqual(abs(data_y[i] - copy_data_y[i]), 5 * std)
        
    def test_n_inliers_less_than_2(self):
        # initiate all arguments
        n_inliers = 1
        data_x = [float("INF")] * n_inliers
        data_y = [float("INF")] * n_inliers
        slope = 0
        intercept = 0
        x_min = 0
        x_max = n_inliers - 1
        # fill data_x, data_y, passing them to make_inliers
        # assume n_iniers < 2 not caught here
        generator.make_inliers(data_x, data_y, n_inliers, slope, intercept,
                              x_min, x_max)
        copy_data_y = data_y.copy()
        # add gaussian noice
        std = 1
        ret = generator.add_gaussian_noise(data_y, n_inliers, std)
        # assertion
        self.assertEqual(ret, -1)


    def test_std_less_than_0(self):
        # initiate all arguments
        n_inliers = 1
        data_x = [float("INF")] * n_inliers
        data_y = [float("INF")] * n_inliers
        slope = 0
        intercept = 0
        x_min = 0
        x_max = n_inliers - 1
        # fill data_x, data_y, passing them to make_inliers
        generator.make_inliers(data_x, data_y, n_inliers, slope, intercept,
                              x_min, x_max)
        copy_data_y = data_y.copy()
        # add gaussian noice
        std = -1
        ret = generator.add_gaussian_noise(data_y, n_inliers, std)
        # assertion
        self.assertEqual(ret, -1)


    def test_std_0(self):
        # initiate all arguments
        n_inliers = 1
        data_x = [float("INF")] * n_inliers
        data_y = [float("INF")] * n_inliers
        slope = 0
        intercept = 0
        x_min = 0
        x_max = n_inliers - 1
        # fill data_x, data_y, passing them to make_inliers
        generator.make_inliers(data_x, data_y, n_inliers, slope, intercept,
                              x_min, x_max)
        copy_data_y = data_y.copy()
        # add gaussian noice
        std = 0
        ret = generator.add_gaussian_noise(data_y, n_inliers, std)
        # assertion
        self.assertEqual(ret, -1)


if __name__ == '__main__':
    unittest.main()
