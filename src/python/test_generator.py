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

import math
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


class TestAddLaplaceNoise(unittest.TestCase):
    """
    Tests happy path and edge cases for add_laplace_noise.
    Happy path:
        all noisy values should be original data_y[i] +/- 8 * scale_noise
    Edge cases:
        n_inliers < 2, should return -1
        scale_noise <= 0, should return -1
        scale_noise == 0, should return -1
    """
    def test_happy_path(self):
        """
        All noisy data_y[i] should be original data_y[i] +/- 8 * scale_noise.
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
        copy_data_y = data_y.copy()
        # add laplace noice
        scale_noise = 1.5
        generator.add_laplace_noise(data_y, n_inliers, scale_noise)
        # assertions
        for i in range(n_inliers):
            self.assertLessEqual(abs(data_y[i] - copy_data_y[i]),
                                 8 * scale_noise)


    def test_n_inliers_less_than_2(self):
        """
        If n_inliers < 2, should return -1.
        """
        # initiate all arguments
        n_inliers = 1
        data_x = [float("INF")] * n_inliers
        data_y = [float("INF")] * n_inliers
        slope = 1
        intercept = 0
        x_min = 0
        x_max = n_inliers - 1
        # fill data_x, data_y, passing them to make_inliers
        generator.make_inliers(data_x, data_y, n_inliers, slope, intercept,
                              x_min, x_max)
        copy_data_y = data_y.copy()
        # add laplace noice
        scale_noise = 1.5
        ret = generator.add_laplace_noise(data_y, n_inliers, scale_noise)
        # assertion
        self.assertEqual(ret, -1)


    def test_scale_noise_zero(self):
        """
        If scale_noise == 0, should return -1.
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
        copy_data_y = data_y.copy()
        # add laplace noice
        scale_noise = 0
        ret = generator.add_laplace_noise(data_y, n_inliers, scale_noise)
        # assertion
        self.assertEqual(ret, -1)


    def test_scale_noise_negative(self):
        """
        If scale_noise < 0, should return -1.
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
        copy_data_y = data_y.copy()
        # add laplace noice
        scale_noise = -1.5
        ret = generator.add_laplace_noise(data_y, n_inliers, scale_noise)
        # assertion
        self.assertEqual(ret, -1)


class TestAddStructuralBias(unittest.TestCase):
    """
    Tests happy paths and edge cases for add_structural_bias.
    Happy paths:
        bias_fn = lambda x: 0,           each data_y[i] unchanged
        bias_fn = lambda x: 1.0,         each data_y[i] increased by 1.0
        bias_fn = lambda x: 0.5 * x,     each data_y[i] increased by 0.5 * data_x[i]
        bias_fn = lambda x: math.sin(x), each data_y[i] increased by sin(data_x[i])
    Edge cases:
        n_inliers < 2, should return -1
        bias_fn is None, should return -1
    """

    def test_zero_bias(self):
        """
        bias_fn = lambda x: 0, data_y should be unchanged.
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
        copy_data_y = data_y.copy()
        # add structural bias
        bias_fn = lambda x: 0
        ret = generator.add_structural_bias(data_y, data_x, n_inliers, bias_fn)
        # assertion
        for i in range(n_inliers):
            self.assertAlmostEqual(abs(data_y[i] - copy_data_y[i]), 0)


    def test_constant_bias(self):
        """
        bias_fn = lambda x: 1.0, all data_y[i] should increase by 1.0.
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
        copy_data_y = data_y.copy()
        # add structural bias
        bias_fn = lambda x: 1.0
        ret = generator.add_structural_bias(data_y, data_x, n_inliers, bias_fn)
        # assertion
        for i in range(n_inliers):
            self.assertAlmostEqual(abs(data_y[i] - copy_data_y[i]), 1)


    def test_linear_bias(self):
        """
        bias_fn = lambda x: 0.5 * x, data_y[i] should increase by 0.5 * data_x[i].
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
        copy_data_y = data_y.copy()
        # add structural bias
        bias_fn = lambda x: 0.5 * x
        ret = generator.add_structural_bias(data_y, data_x, n_inliers, bias_fn)
        # assertion
        for i in range(n_inliers):
            self.assertAlmostEqual(abs(data_y[i] - copy_data_y[i]),
                                   abs(0.5 * data_x[i]))


    def test_periodic_bias(self):
        """
        bias_fn = lambda x: math.sin(x), data_y[i] should increase by sin(data_x[i]).
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
        copy_data_y = data_y.copy()
        # add structural bias
        bias_fn = lambda x: math.sin(x)
        ret = generator.add_structural_bias(data_y, data_x, n_inliers, bias_fn)
        # assertion
        for i in range(n_inliers):
            self.assertAlmostEqual(abs(data_y[i] - copy_data_y[i]),
                                   abs(math.sin(data_x[i])))



    def test_n_inliers_less_than_2(self):
        """
        If n_inliers < 2, should return -1.
        """
        # initiate all arguments
        n_inliers = 1
        data_x = [float("INF")] * n_inliers
        data_y = [float("INF")] * n_inliers
        slope = 1
        intercept = 0
        x_min = 0
        x_max = n_inliers - 1
        # fill data_x, data_y, passing them to make_inliers
        generator.make_inliers(data_x, data_y, n_inliers, slope, intercept,
                              x_min, x_max)
        copy_data_y = data_y.copy()
        # add structural bias
        bias_fn = lambda x: 1
        ret = generator.add_structural_bias(data_y, data_x, n_inliers, bias_fn)
        # assertion
        self.assertEqual(ret, -1)


    def test_bias_fn_none(self):
        """
        If bias_fn is None, should return -1.
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
        copy_data_y = data_y.copy()
        # add structural bias
        bias_fn = None
        ret = generator.add_structural_bias(data_y, data_x, n_inliers, bias_fn)
        # assertion
        self.assertEqual(ret, -1)


class TestAddOutliers(unittest.TestCase):
    """
    Tests happy paths and edge cases for add_outliers.
    Happy paths:
        n_outliers points added starting at index n_inliers
        all added x values within [x_min, x_max]
        all added y values within [y_min, y_max]
        total length of data_x == n_inliers + n_outliers
    Special case:
        n_outliers == 0, should return 0, data unchanged
    Edge cases:
        n_outliers < 0, should return -1
        n_inliers + n_outliers < 2, should return -1
        x_min == x_max, should return -1
        y_min == y_max, should return -1
    """


    def test_happy_path(self):
        """
        n_outliers points added starting at index n_inliers.
        Tests:
            All added x within [x_min, x_max].
            All added y within [y_min, y_max].
            Total length == n_inliers + n_outliers.
        """
        # initiate all arguments
        n_inliers = 100
        data_x = [float("INF")] * n_inliers
        data_y = [float("INF")] * n_inliers
        slope = 1
        intercept = 0
        x_min = 0
        x_max = n_inliers - 1
        # fill data_x, data_y, passing them to make_inliers
        generator.make_inliers(data_x, data_y, n_inliers, slope, intercept,
                              x_min, x_max)
        copy_data_x = data_x.copy()
        copy_data_y = data_y.copy()
        # add outliers
        n_outliers = n_inliers // 10
        y_min = min(data_y)
        y_max = max(data_y)
        ret = generator.add_outliers(data_x, data_y, n_inliers, n_outliers,
                                     x_min, x_max, y_min, y_max)
        # assertions
        self.assertEqual(len(data_y), n_inliers + n_outliers)
        for i in range(n_inliers + n_outliers):
            if i < n_inliers:  # test original data is unchanged
                self.assertAlmostEqual(data_x[i], copy_data_x[i])
                self.assertAlmostEqual(data_y[i], copy_data_y[i])
                continue
            # test added data is from the space x_min, y_min, x_max, y_max
            self.assertLessEqual(data_x[i], x_max)
            self.assertGreaterEqual(data_x[i], x_min)
            self.assertLessEqual(data_y[i], y_max)
            self.assertGreaterEqual(data_y[i], y_min)


    def test_zero_outliers(self):
        """
        n_outliers == 0, data_x and data_y should be unchanged, return 0.
        """
        # initiate all arguments
        n_inliers = 100
        data_x = [float("INF")] * n_inliers
        data_y = [float("INF")] * n_inliers
        slope = 1
        intercept = 0
        x_min = 0
        x_max = n_inliers - 1
        # fill data_x, data_y, passing them to make_inliers
        generator.make_inliers(data_x, data_y, n_inliers, slope, intercept,
                              x_min, x_max)
        copy_data_x = data_x.copy()
        copy_data_y = data_y.copy()
        # add outliers
        n_outliers = 0
        y_min = min(data_y)
        y_max = max(data_y)
        ret = generator.add_outliers(data_x, data_y, n_inliers, n_outliers,
                                     x_min, x_max, y_min, y_max)
        # assertions
        self.assertEqual(len(data_y), n_inliers + n_outliers)
        for i in range(n_inliers + n_outliers):
            if i < n_inliers:  # test original data is unchanged
                self.assertAlmostEqual(data_x[i], copy_data_x[i])
                self.assertAlmostEqual(data_y[i], copy_data_y[i])
                continue
            # test added data is from the space x_min, y_min, x_max, y_max
            self.assertLessEqual(data_x[i], x_max)
            self.assertGreaterEqual(data_x[i], x_min)
            self.assertLessEqual(data_y[i], y_max)
            self.assertGreaterEqual(data_y[i], y_min)


    def test_n_outliers_negative(self):
        """
        If n_outliers < 0, should return -1.
        """
        # initiate all arguments
        n_inliers = 100
        data_x = [float("INF")] * n_inliers
        data_y = [float("INF")] * n_inliers
        slope = 1
        intercept = 0
        x_min = 0
        x_max = n_inliers - 1
        # fill data_x, data_y, passing them to make_inliers
        generator.make_inliers(data_x, data_y, n_inliers, slope, intercept,
                              x_min, x_max)
        copy_data_x = data_x.copy()
        copy_data_y = data_y.copy()
        # add outliers
        n_outliers = -10
        y_min = min(data_y)
        y_max = max(data_y)
        ret = generator.add_outliers(data_x, data_y, n_inliers, n_outliers,
                                     x_min, x_max, y_min, y_max)
        # assertions
        self.assertEqual(ret, -1)


    def test_final_n_less_than_2(self):
        """
        If n_inliers + n_outliers < 2, should return -1.
        """
        # initiate all arguments for add_outliers
        data_y = []
        data_x = []
        n_inliers = 0
        n_outliers = 1
        # min/max incorrect, but kept to test the final_n_less_than_2 in isolation
        x_min = 0
        x_max = 1
        y_min = 0
        y_max = 1
        ret = generator.add_outliers(data_x, data_y, n_inliers, n_outliers,
                                     x_min, x_max, y_min, y_max)
        # assertions
        self.assertEqual(ret, -1)


    def test_x_min_equals_x_max(self):
        """
        If x_min == x_max, should return -1.
        """
        # initiate all arguments
        n_inliers = 100
        data_x = [float("INF")] * n_inliers
        data_y = [float("INF")] * n_inliers
        slope = 1
        intercept = 0
        x_min = 0
        x_max = n_inliers - 1
        # fill data_x, data_y, passing them to make_inliers
        generator.make_inliers(data_x, data_y, n_inliers, slope, intercept,
                              x_min, x_max)
        copy_data_x = data_x.copy()
        copy_data_y = data_y.copy()
        # add outliers
        n_outliers = -10
        y_min = min(data_y)
        y_max = max(data_y)
        x_max = n_inliers - 1 # override x_max for the test
        ret = generator.add_outliers(data_x, data_y, n_inliers, n_outliers,
                                     x_min, x_max, y_min, y_max)
         # assertions
        self.assertEqual(ret, -1)


    def test_y_min_equals_y_max(self):
        """
        If y_min == y_max, should return -1.
        """
        # initiate all arguments
        n_inliers = 100
        data_x = [float("INF")] * n_inliers
        data_y = [float("INF")] * n_inliers
        slope = 1
        intercept = 0
        x_min = 0
        x_max = n_inliers - 1
        # fill data_x, data_y, passing them to make_inliers
        generator.make_inliers(data_x, data_y, n_inliers, slope, intercept,
                              x_min, x_max)
        copy_data_x = data_x.copy()
        copy_data_y = data_y.copy()
        # add outliers
        n_outliers = -10
        y_min = min(data_y)
        y_max = y_min       # for testsing
        ret = generator.add_outliers(data_x, data_y, n_inliers, n_outliers,
                                     x_min, x_max, y_min, y_max)
         # assertions
        self.assertEqual(ret, -1)


if __name__ == '__main__':
    unittest.main()
