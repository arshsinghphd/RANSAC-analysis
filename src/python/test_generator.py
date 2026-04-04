"""
Tests for generator.py functions:
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
    Tests happy paths, combinations of:
        Positive and negative slopes
        Positive and negative intercepts

    Special cases:
        slope = 0, intercept = 0 (flat line)

    Edge cases:
        n = 0 should return -1
        n = 1 should return -1
        x_min = x_max should return -1
    """

    def setUp(self):
        self.n = 10
        self.x_min = 0
        self.x_max = self.n - 1

    def _make_data(self, n=None):
        n = n if n else self.n
        data_x = [float("inf")] * n
        data_y = [float("inf")] * n
        return data_x, data_y

    def _assert_line(self, data_x, data_y, slope, intercept, n=None):
        n = n if n else self.n
        exp_x = [float(self.x_min + i * (self.x_max - self.x_min) / (n - 1))
                 for i in range(n)]
        exp_y = [slope * x + intercept for x in exp_x]
        for i in range(n):
            self.assertAlmostEqual(data_x[i], exp_x[i])
            self.assertAlmostEqual(data_y[i], exp_y[i])


    def test_unit_slope_zero_intercept(self):
        """
        slope = 1, intercept = 0.
        data_x = [0, ..., n-1], data_y = [0, ..., n-1].
        """
        data_x, data_y = self._make_data()
        generator.make_inliers(data_x, data_y, self.n, 1, 0,
                               self.x_min, self.x_max)
        self._assert_line(data_x, data_y, 1, 0)


    def test_negative_unit_slope_zero_intercept(self):
        """
        slope = -1, intercept = n - 1.
        data_x = [0, ..., n-1], data_y = [n-1, ..., 0].
        """
        data_x, data_y = self._make_data()
        generator.make_inliers(data_x, data_y, self.n, -1, self.n - 1,
                               self.x_min, self.x_max)
        self._assert_line(data_x, data_y, -1, self.n - 1)


    def test_float_slope_zero_intercept(self):
        """
        slope = 0.5, intercept = 0.
        data_x = [0, ..., n-1], data_y = [0, 0.5, ..., (n-1)/2].
        """
        data_x, data_y = self._make_data()
        generator.make_inliers(data_x, data_y, self.n, 0.5, 0,
                               self.x_min, self.x_max)
        self._assert_line(data_x, data_y, 0.5, 0)


    def test_unit_slope_negative_intercept(self):
        """
        slope = 1, intercept = -n + 1.
        data_x = [0, ..., n-1], data_y = [-n+1, ..., 0].
        """
        data_x, data_y = self._make_data()
        generator.make_inliers(data_x, data_y, self.n, 1, -self.n + 1,
                               self.x_min, self.x_max)
        self._assert_line(data_x, data_y, 1, -self.n + 1)


    def test_zero_slope_zero_intercept(self):
        """
        slope = 0, intercept = 0. Special case: flat line at y = 0.
        """
        data_x, data_y = self._make_data()
        generator.make_inliers(data_x, data_y, self.n, 0, 0,
                               self.x_min, self.x_max)
        self._assert_line(data_x, data_y, 0, 0)


    def test_n_less_than_2(self):
        """
        n = 1, should return -1.
        """
        data_x, data_y = self._make_data(n=1)
        ret = generator.make_inliers(data_x, data_y, 1, 1, 0,
                                     self.x_min, self.x_max)
        self.assertEqual(ret, -1)


    def test_x_min_equals_x_max(self):
        """
        x_min == x_max, should return -1.
        """
        data_x, data_y = self._make_data()
        ret = generator.make_inliers(data_x, data_y, self.n, 1, 0, 0, 0)
        self.assertEqual(ret, -1)


class TestAddGaussianNoise(unittest.TestCase):
    """
    Tests happy paths and edge cases for add_gaussian_noise.
    Happy paths:
        all noisy data_y[i] within original data_y[i] plus or minus 5 * std
    Edge cases:
        n_inliers < 2, should return -1
        std <= 0, should return -1
        std == 0, should return -1
    """

    def setUp(self):
        self.n = 10
        self.x_min = 0
        self.x_max = self.n - 1
        self.data_x = [float("inf")] * self.n
        self.data_y = [float("inf")] * self.n
        generator.make_inliers(self.data_x, self.data_y, self.n, 1, 0,
                               self.x_min, self.x_max)

    def _assert_within_bound(self, original_y, noisy_y, bound):
        for i in range(len(original_y)):
            self.assertLessEqual(abs(noisy_y[i] - original_y[i]), bound)


    def test_happy_path_int_std(self):
        """
        std = 1. All noisy data_y[i] within original plus or minus 5 * std.
        """
        std = 1
        copy_y = self.data_y.copy()
        generator.add_gaussian_noise(self.data_y, self.n, std)
        self._assert_within_bound(copy_y, self.data_y, 5 * std)


    def test_happy_path_float_std(self):
        """
        std = (max - min) / 6. All noisy data_y[i] within original plus or minus 5 * std.
        """
        std = (max(self.data_y) - min(self.data_y)) / 6
        copy_y = self.data_y.copy()
        generator.add_gaussian_noise(self.data_y, self.n, std)
        self._assert_within_bound(copy_y, self.data_y, 5 * std)


    def test_n_inliers_less_than_2(self):
        """
        n_inliers = 1, should return -1.
        """
        ret = generator.add_gaussian_noise(self.data_y, 1, 1.0)
        self.assertEqual(ret, -1)


    def test_std_negative(self):
        """
        std = -1, should return -1.
        """
        ret = generator.add_gaussian_noise(self.data_y, self.n, -1)
        self.assertEqual(ret, -1)


    def test_std_zero(self):
        """
        std = 0, should return -1.
        """
        ret = generator.add_gaussian_noise(self.data_y, self.n, 0)
        self.assertEqual(ret, -1)


class TestAddLaplaceNoise(unittest.TestCase):
    """
    Tests happy path and edge cases for add_laplace_noise.
    Happy path:
        all noisy data_y[i] within original data_y[i] plus or minus 8 * scale_noise
    Edge cases:
        n_inliers < 2, should return -1
        scale_noise <= 0, should return -1
    """

    def setUp(self):
        self.n = 10
        self.x_min = 0
        self.x_max = self.n - 1
        self.data_x = [float("inf")] * self.n
        self.data_y = [float("inf")] * self.n
        generator.make_inliers(self.data_x, self.data_y, self.n, 1, 0,
                               self.x_min, self.x_max)


    def test_happy_path(self):
        """
        scale = 1.5. All noisy data_y[i] within original plus or minus 8 * scale_noise.
        """
        scale = 1.5
        copy_y = self.data_y.copy()
        generator.add_laplace_noise(self.data_y, self.n, scale)
        for i in range(self.n):
            self.assertLessEqual(abs(self.data_y[i] - copy_y[i]), 8 * scale)


    def test_n_inliers_less_than_2(self):
        """
        n_inliers = 1, should return -1.
        """
        ret = generator.add_laplace_noise(self.data_y, 1, 1.5)
        self.assertEqual(ret, -1)


    def test_scale_zero(self):
        """
        scale_noise = 0, should return -1.
        """
        ret = generator.add_laplace_noise(self.data_y, self.n, 0)
        self.assertEqual(ret, -1)


    def test_scale_negative(self):
        """
        scale_noise = -1.5, should return -1.
        """
        ret = generator.add_laplace_noise(self.data_y, self.n, -1.5)
        self.assertEqual(ret, -1)


class TestAddStructuralBias(unittest.TestCase):
    """
    Tests happy paths and edge cases for add_structural_bias.
    Happy paths:
        bias_fn = lambda x: 0,           data_y unchanged
        bias_fn = lambda x: 1.0,         data_y[i] increased by 1.0
        bias_fn = lambda x: 0.5 * x,     data_y[i] increased by 0.5 * data_x[i]
        bias_fn = lambda x: math.sin(x), data_y[i] increased by sin(data_x[i])
    Edge cases:
        n_inliers < 2, should return -1
        bias_fn is None, should return -1
    """

    def setUp(self):
        self.n = 10
        self.x_min = 0
        self.x_max = self.n - 1
        self.data_x = [float("inf")] * self.n
        self.data_y = [float("inf")] * self.n
        generator.make_inliers(self.data_x, self.data_y, self.n, 1, 0,
                               self.x_min, self.x_max)

    def _assert_bias(self, copy_y, bias_fn):
        for i in range(self.n):
            self.assertAlmostEqual(
                self.data_y[i] - copy_y[i],
                bias_fn(self.data_x[i])
            )


    def test_zero_bias(self):
        """
        bias_fn = lambda x: 0, data_y should be unchanged.
        """
        copy_y = self.data_y.copy()
        generator.add_structural_bias(self.data_y, self.data_x, self.n,
                                      lambda x: 0)
        self._assert_bias(copy_y, lambda x: 0)


    def test_constant_bias(self):
        """
        bias_fn = lambda x: 1.0, all data_y[i] increased by 1.0.
        """
        copy_y = self.data_y.copy()
        generator.add_structural_bias(self.data_y, self.data_x, self.n,
                                      lambda x: 1.0)
        self._assert_bias(copy_y, lambda x: 1.0)


    def test_linear_bias(self):
        """
        bias_fn = lambda x: 0.5 * x, data_y[i] increased by 0.5 * data_x[i].
        """
        copy_y = self.data_y.copy()
        generator.add_structural_bias(self.data_y, self.data_x, self.n,
                                      lambda x: 0.5 * x)
        self._assert_bias(copy_y, lambda x: 0.5 * x)


    def test_periodic_bias(self):
        """
        bias_fn = lambda x: math.sin(x), data_y[i] increased by sin(data_x[i]).
        """
        copy_y = self.data_y.copy()
        generator.add_structural_bias(self.data_y, self.data_x, self.n,
                                      lambda x: math.sin(x))
        self._assert_bias(copy_y, lambda x: math.sin(x))


    def test_n_inliers_less_than_2(self):
        """
        n_inliers = 1, should return -1.
        """
        ret = generator.add_structural_bias(self.data_y, self.data_x, 1,
                                            lambda x: 1)
        self.assertEqual(ret, -1)


    def test_bias_fn_none(self):
        """
        bias_fn = None, should return -1.
        """
        ret = generator.add_structural_bias(self.data_y, self.data_x,
                                            self.n, None)
        self.assertEqual(ret, -1)


class TestAddOutliers(unittest.TestCase):
    """
    Tests happy paths and edge cases for add_outliers.
    Happy paths:
        n_outliers points appended starting at index n_inliers
        all added x within [x_min, x_max]
        all added y within [y_min, y_max]
        total length equals n_inliers + n_outliers
    Special case:
        n_outliers == 0, data unchanged, return 0
    Edge cases:
        n_outliers < 0, should return -1
        n_inliers + n_outliers < 2, should return -1
        x_min == x_max, should return -1
        y_min == y_max, should return -1
    """

    def setUp(self):
        self.n = 100
        self.x_min = 0
        self.x_max = self.n - 1
        self.data_x = [float("inf")] * self.n
        self.data_y = [float("inf")] * self.n
        generator.make_inliers(self.data_x, self.data_y, self.n, 1, 0,
                               self.x_min, self.x_max)
        self.y_min = min(self.data_y)
        self.y_max = max(self.data_y)

    def _assert_outliers_in_range(self, n_outliers, orig_x, orig_y):
        self.assertEqual(len(self.data_x), self.n + n_outliers)
        for i in range(self.n):
            self.assertAlmostEqual(self.data_x[i], orig_x[i])
            self.assertAlmostEqual(self.data_y[i], orig_y[i])
        for i in range(self.n, self.n + n_outliers):
            self.assertGreaterEqual(self.data_x[i], self.x_min)
            self.assertLessEqual(self.data_x[i], self.x_max)
            self.assertGreaterEqual(self.data_y[i], self.y_min)
            self.assertLessEqual(self.data_y[i], self.y_max)


    def test_happy_path(self):
        """
        10 outliers appended. All within range, inliers unchanged.
        """
        n_out = self.n // 10
        orig_x = self.data_x.copy()
        orig_y = self.data_y.copy()
        generator.add_outliers(self.data_x, self.data_y, self.n, n_out,
                               self.x_min, self.x_max, self.y_min, self.y_max)
        self._assert_outliers_in_range(n_out, orig_x, orig_y)


    def test_zero_outliers(self):
        """
        n_outliers = 0, data unchanged, return 0.
        """
        orig_x = self.data_x.copy()
        orig_y = self.data_y.copy()
        ret = generator.add_outliers(self.data_x, self.data_y, self.n, 0,
                                     self.x_min, self.x_max,
                                     self.y_min, self.y_max)
        self.assertEqual(ret, 0)
        self._assert_outliers_in_range(0, orig_x, orig_y)


    def test_n_outliers_negative(self):
        """
        n_outliers = -10, should return -1.
        """
        ret = generator.add_outliers(self.data_x, self.data_y, self.n, -10,
                                     self.x_min, self.x_max,
                                     self.y_min, self.y_max)
        self.assertEqual(ret, -1)


    def test_final_n_less_than_2(self):
        """
        n_inliers = 0, n_outliers = 1, total less than 2, should return -1.
        """
        ret = generator.add_outliers([], [], 0, 1,
                                     self.x_min, self.x_max,
                                     self.y_min, self.y_max)
        self.assertEqual(ret, -1)


    def test_x_min_equals_x_max(self):
        """
        x_min == x_max, should return -1.
        """
        ret = generator.add_outliers(self.data_x, self.data_y, self.n, 10,
                                     5, 5, self.y_min, self.y_max)
        self.assertEqual(ret, -1)


    def test_y_min_equals_y_max(self):
        """
        y_min == y_max, should return -1.
        """
        ret = generator.add_outliers(self.data_x, self.data_y, self.n, 10,
                                     self.x_min, self.x_max, 5, 5)
        self.assertEqual(ret, -1)


if __name__ == '__main__':
    unittest.main()
