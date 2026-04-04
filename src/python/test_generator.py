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
        n = 1 should return -1
        x_min = x_max should return -1
    """

    def setUp(self):
        self.n = 10
        self.x_min = 0
        self.x_max = self.n - 1

    def _make_line(self, points_x, points_y, slope, intercept, n=None):
        n = n if n else self.n
        generator.make_inliers(points_x, points_y, n, slope, intercept,
                               self.x_min, self.x_max)

    def _assert_line(self, points_x, points_y, slope, intercept, n=None):
        n = n if n else self.n
        exp_x = [float(self.x_min + i * (self.x_max - self.x_min) / (n - 1))
                 for i in range(n)]
        exp_y = [slope * x + intercept for x in exp_x]
        for i in range(n):
            self.assertAlmostEqual(points_x[i], exp_x[i])
            self.assertAlmostEqual(points_y[i], exp_y[i])


    def test_unit_slope_zero_intercept(self):
        """
        slope = 1, intercept = 0.
        expected_x = [0, ..., n-1], expected_y = [0, ..., n-1].
        """
        points_x = [float("inf")] * self.n
        points_y = [float("inf")] * self.n
        self._make_line(points_x, points_y, 1, 0)
        self._assert_line(points_x, points_y, 1, 0)


    def test_negative_unit_slope(self):
        """
        slope = -1, intercept = n - 1.
        points_x = [0, ..., n-1], points_y = [n-1, ..., 0].
        """
        points_x = [float("inf")] * self.n
        points_y = [float("inf")] * self.n
        self._make_line(points_x, points_y, -1, self.n - 1)
        self._assert_line(points_x, points_y, -1, self.n - 1)


    def test_float_slope_zero_intercept(self):
        """
        slope = 0.5, intercept = 0.
        points_x = [0, ..., n-1], points_y = [0, 0.5, ..., (n-1)/2].
        """
        points_x = [float("inf")] * self.n
        points_y = [float("inf")] * self.n
        self._make_line(points_x, points_y, 0.5, 0)
        self._assert_line(points_x, points_y, 0.5, 0)


    def test_unit_slope_negative_intercept(self):
        """
        slope = 1, intercept = -n + 1.
        points_x = [0, ..., n-1], points_y = [-n+1, ..., 0].
        """
        points_x = [float("inf")] * self.n
        points_y = [float("inf")] * self.n
        self._make_line(points_x, points_y, 1, -self.n + 1)
        self._assert_line(points_x, points_y, 1, -self.n + 1)


    def test_zero_slope_zero_intercept(self):
        """
        slope = 0, intercept = 0. Special case: flat line at y = 0.
        """
        points_x = [float("inf")] * self.n
        points_y = [float("inf")] * self.n
        self._make_line(points_x, points_y, 0, 0)
        self._assert_line(points_x, points_y, 0, 0)


    def test_n_less_than_2(self):
        """
        n = 1, should return -1.
        """
        points_x = [float("inf")] * 1
        points_y = [float("inf")] * 1
        ret = generator.make_inliers(points_x, points_y, 1, 1, 0,
                                     self.x_min, self.x_max)
        self.assertEqual(ret, -1)


    def test_x_min_equals_x_max(self):
        """
        x_min == x_max, should return -1.
        """
        points_x = [float("inf")] * self.n
        points_y = [float("inf")] * self.n
        ret = generator.make_inliers(points_x, points_y, self.n, 1, 0, 0, 0)
        self.assertEqual(ret, -1)


class TestAddGaussianNoise(unittest.TestCase):
    """
    Tests happy paths and edge cases for add_gaussian_noise.
    Happy paths:
        all noisy points_y[i] within original points_y[i] plus or minus 5 * std
    Edge cases:
        n_inliers < 2, should return -1
        std <= 0, should return -1
        std == 0, should return -1
    """

    def setUp(self):
        self.n = 10
        self.x_min = 0
        self.x_max = self.n - 1
        self.points_x = [float("inf")] * self.n
        self.points_y = [float("inf")] * self.n
        generator.make_inliers(self.points_x, self.points_y, self.n, 1, 0,
                               self.x_min, self.x_max)

    def _assert_within_bound(self, original_y, noisy_y, n, bound):
        for i in range(n):
            self.assertLessEqual(abs(noisy_y[i] - original_y[i]), bound)


    def test_happy_path_int_std(self):
        """
        std = 1. All noisy points_y[i] within original plus or minus 5 * std.
        """
        std = 1
        copy_y = self.points_y.copy()
        generator.add_gaussian_noise(self.points_y, self.n, std)
        self._assert_within_bound(copy_y, self.points_y, self.n, 5 * std)


    def test_happy_path_float_std(self):
        """
        std = (max - min) / 6. All noisy points_y[i] within original plus or minus 5 * std.
        """
        std = (max(self.points_y) - min(self.points_y)) / 6
        copy_y = self.points_y.copy()
        generator.add_gaussian_noise(self.points_y, self.n, std)
        self._assert_within_bound(copy_y, self.points_y, self.n, 5 * std)


    def test_n_inliers_less_than_2(self):
        """
        n_inliers = 1, should return -1.
        """
        ret = generator.add_gaussian_noise(self.points_y, 1, 1.0)
        self.assertEqual(ret, -1)


    def test_std_negative(self):
        """
        std = -1, should return -1.
        """
        ret = generator.add_gaussian_noise(self.points_y, self.n, -1)
        self.assertEqual(ret, -1)


    def test_std_zero(self):
        """
        std = 0, should return -1.
        """
        ret = generator.add_gaussian_noise(self.points_y, self.n, 0)
        self.assertEqual(ret, -1)


class TestAddLaplaceNoise(unittest.TestCase):
    """
    Tests happy path and edge cases for add_laplace_noise.
    Happy path:
        all noisy points_y[i] within original points_y[i] plus or minus 8 * scale_noise
    Edge cases:
        n_inliers < 2, should return -1
        scale_noise <= 0, should return -1
    """

    def setUp(self):
        self.n = 10
        self.x_min = 0
        self.x_max = self.n - 1
        self.points_x = [float("inf")] * self.n
        self.points_y = [float("inf")] * self.n
        generator.make_inliers(self.points_x, self.points_y, self.n, 1, 0,
                               self.x_min, self.x_max)


    def test_happy_path(self):
        """
        scale = 1.5. All noisy points_y[i] within original plus or minus 8 * scale.
        """
        scale = 1.5
        copy_y = self.points_y.copy()
        generator.add_laplace_noise(self.points_y, self.n, scale)
        for i in range(self.n):
            self.assertLessEqual(abs(self.points_y[i] - copy_y[i]), 8 * scale)


    def test_n_inliers_less_than_2(self):
        """
        n_inliers = 1, should return -1.
        """
        ret = generator.add_laplace_noise(self.points_y, 1, 1.5)
        self.assertEqual(ret, -1)


    def test_scale_zero(self):
        """
        scale_noise = 0, should return -1.
        """
        ret = generator.add_laplace_noise(self.points_y, self.n, 0)
        self.assertEqual(ret, -1)


    def test_scale_negative(self):
        """
        scale_noise = -1.5, should return -1.
        """
        ret = generator.add_laplace_noise(self.points_y, self.n, -1.5)
        self.assertEqual(ret, -1)


class TestAddStructuralBias(unittest.TestCase):
    """
    Tests happy paths and edge cases for add_structural_bias.
    Happy paths:
        bias_fn = lambda x: 0,           points_y unchanged
        bias_fn = lambda x: 1.0,         points_y[i] increased by 1.0
        bias_fn = lambda x: 0.5 * x,     points_y[i] increased by 0.5 * points_x[i]
        bias_fn = lambda x: math.sin(x), points_y[i] increased by sin(points_x[i])
    Edge cases:
        n_inliers < 2, should return -1
        bias_fn is None, should return -1
    """

    def setUp(self):
        self.n = 10
        self.x_min = 0
        self.x_max = self.n - 1
        self.points_x = [float("inf")] * self.n
        self.points_y = [float("inf")] * self.n
        generator.make_inliers(self.points_x, self.points_y, self.n, 1, 0,
                               self.x_min, self.x_max)

    def _assert_bias(self, copy_y, bias_fn, n=None):
        n = n if n else self.n
        for i in range(n):
            self.assertAlmostEqual(
                self.points_y[i] - copy_y[i],
                bias_fn(self.points_x[i])
            )


    def test_zero_bias(self):
        """
        bias_fn = lambda x: 0, points_y should be unchanged.
        """
        copy_y = self.points_y.copy()
        generator.add_structural_bias(self.points_y, self.points_x, self.n,
                                      lambda x: 0)
        self._assert_bias(copy_y, lambda x: 0)


    def test_constant_bias(self):
        """
        bias_fn = lambda x: 1.0, all points_y[i] increased by 1.0.
        """
        copy_y = self.points_y.copy()
        generator.add_structural_bias(self.points_y, self.points_x, self.n,
                                      lambda x: 1.0)
        self._assert_bias(copy_y, lambda x: 1.0)


    def test_linear_bias(self):
        """
        bias_fn = lambda x: 0.5 * x, points_y[i] increased by 0.5 * points_x[i].
        """
        copy_y = self.points_y.copy()
        generator.add_structural_bias(self.points_y, self.points_x, self.n,
                                      lambda x: 0.5 * x)
        self._assert_bias(copy_y, lambda x: 0.5 * x)


    def test_periodic_bias(self):
        """
        bias_fn = lambda x: math.sin(x), points_y[i] increased by sin(points_x[i]).
        """
        copy_y = self.points_y.copy()
        generator.add_structural_bias(self.points_y, self.points_x, self.n,
                                      lambda x: math.sin(x))
        self._assert_bias(copy_y, lambda x: math.sin(x))


    def test_n_inliers_less_than_2(self):
        """
        n_inliers = 1, should return -1.
        """
        ret = generator.add_structural_bias(self.points_y, self.points_x, 1,
                                            lambda x: 1)
        self.assertEqual(ret, -1)


    def test_bias_fn_none(self):
        """
        bias_fn = None, should return -1.
        """
        ret = generator.add_structural_bias(self.points_y, self.points_x,
                                            self.n, None)
        self.assertEqual(ret, -1)


class TestAddOutliers(unittest.TestCase):
    """
    Tests happy paths and edge cases for add_outliers.

    add_outliers add points that are true classification errors
    by placing outliers outside the inlier band defined by the true model.
    The inlier band is:
        (slope * x + intercept) +/- 2 * noise_std * sqrt(1 + slope^2).

    Happy paths:
        n_outliers points appended, all outside the inlier band
        total length equals n_inliers + n_outliers
        inlier points unchanged after call

    Special case:
        n_outliers == 0, data unchanged, return 0

    Edge cases:
        n_inliers < 0,              should return -1
        n_outliers < 0,             should return -1
        n_inliers + n_outliers < 2, should return -1
        noise_std <= 0,             should return -1
    """

    def setUp(self):
        self.n = 100
        self.x_min = 0
        self.x_max = self.n - 1
        self.slope = 1.0
        self.intercept = 0.0
        self.noise_std = 0.5
        self.points_x = [float("inf")] * self.n
        self.points_y = [float("inf")] * self.n
        generator.make_inliers(self.points_x, self.points_y, self.n,
                               self.slope, self.intercept,
                               self.x_min, self.x_max)
        generator.add_gaussian_noise(self.points_y, self.n, self.noise_std)

    def _assert_outliers_outside_band(self, n_outliers, orig_x, orig_y):
        """
        Asserts total length is correct, inliers are unchanged, and every
        added point lies outside the inlier band.
        """
        self.assertEqual(len(self.points_x), self.n + n_outliers)
        for i in range(self.n):
            self.assertAlmostEqual(self.points_x[i], orig_x[i])
            self.assertAlmostEqual(self.points_y[i], orig_y[i])
        inlier_band = 2 * self.noise_std * math.sqrt(
            1 + self.slope * self.slope)
        for i in range(self.n, self.n + n_outliers):
            x = self.points_x[i]
            y = self.points_y[i]
            dist = abs(self.slope * x - y + self.intercept) / math.sqrt(
                1 + self.slope * self.slope)
            self.assertGreater(dist, inlier_band)


    def test_happy_path(self):
        """
        10% outliers appended. All outside inlier band, inliers unchanged.
        """
        n_out = self.n // 10
        orig_x = self.points_x.copy()
        orig_y = self.points_y.copy()
        # add outliers
        generator.add_outliers(self.points_x, self.points_y, self.n, n_out,
                               self.slope, self.intercept, self.noise_std)
        # assertions
        self._assert_outliers_outside_band(n_out, orig_x, orig_y)


    def test_zero_outliers(self):
        """
        n_outliers = 0, data unchanged, return 0.
        """
        orig_x = self.points_x.copy()
        orig_y = self.points_y.copy()
        # add outliers
        ret = generator.add_outliers(self.points_x, self.points_y, self.n, 0,
                                     self.slope, self.intercept, self.noise_std)
        # assertions
        self.assertEqual(ret, 0)
        self._assert_outliers_outside_band(0, orig_x, orig_y)


    def test_n_outliers_negative(self):
        """
        n_outliers = -10, should return -1.
        """
        # add outliers
        ret = generator.add_outliers(self.points_x, self.points_y, self.n,
                                     -10, self.slope, self.intercept,
                                     self.noise_std)
        # assertions
        self.assertEqual(ret, -1)


    def test_n_inliers_negative(self):
        """
        n_inliers = -1, should return -1.
        """
        # add outliers
        ret = generator.add_outliers(self.points_x, self.points_y, -1, 10,
                                     self.slope, self.intercept, self.noise_std)
        #assertions
        self.assertEqual(ret, -1)


    def test_final_n_less_than_2(self):
        """
        n_inliers = 0, n_outliers = 1, total less than 2, should return -1.
        """
        # add outliers
        ret = generator.add_outliers([], [], 0, 1,
                                     self.slope, self.intercept, self.noise_std)
        # assertions
        self.assertEqual(ret, -1)


    def test_noise_std_zero(self):
        """
        noise_std = 0, should return -1.
        """
        # add outliers
        ret = generator.add_outliers(self.points_x, self.points_y, self.n,
                                     10, self.slope, self.intercept, 0)
        # assertions
        self.assertEqual(ret, -1)


    def test_noise_std_negative(self):
        """
        noise_std = -0.5, should return -1.
        """
        # add outliers
        ret = generator.add_outliers(self.points_x, self.points_y, self.n,
                                     10, self.slope, self.intercept, -0.5)
        # assertions
        self.assertEqual(ret, -1)


if __name__ == '__main__':
    unittest.main()
