"""
Tests for model.py functions: fit_line and points_to_line_distances.

fit_line estimates slope and intercept from n_points using least squares,
storing results at position pos in list_slopes and list_intercepts.

points_to_line_distances computes perpendicular distance from each point
to a line defined by slope and intercept, storing results in distances.

Run from terminal as follows:
    $ python -m unittest test_model.py
"""

import generator
import model

import math
import unittest


class TestFitLine(unittest.TestCase):
    """
    Tests for fit_line which estimates slope and intercept from n_points
    using least squares, storing results at position pos in list_slopes
    and list_intercepts.

    Happy paths:
        slope = 1, intercept = 0, n_points = 10
            exact recovery on clean data
        slope = -1, intercept = 5, n_points = 10
            exact recovery with negative slope
        slope = 0, intercept = 3, n_points = 10
            exact recovery for flat line
        slope = 0.5, intercept = 0, n_points = 10
            exact recovery for fractional slope
        slope = 1, intercept = 0, n_points = 2
            exact recovery at minimum sample size

    Edge cases:
        pos < 0             should return -1
        n_points < 2        should return -1
        all x values equal  should return -1 (vertical line, slope undefined)
    """

    def setUp(self):
        self.n_points = 10
        self.x_min = 0
        self.x_max = self.n_points - 1
        self.pos = 3
        self.list_slopes = [float("inf")] * (self.pos + 1)
        self.list_intercepts = [float("inf")] * (self.pos + 1)
        self.points_x = [float("inf")] * self.n_points
        self.points_y = [float("inf")] * self.n_points


    def _make_line(self, slope, intercept, n_points=None):
        n = n_points if n_points else self.n_points
        points_x = [float("inf")] * n
        points_y = [float("inf")] * n
        generator.make_inliers(points_x, points_y, n, slope, intercept,
                               self.x_min, self.x_max)
        return points_x, points_y


    def test_unit_slope_zero_intercept(self):
        """
        slope = 1, intercept = 0, n_points = 10, clean data.
        fit_line should recover slope = 1, intercept = 0 exactly.
        """
        slope, intercept = 1, 0
        points_x, points_y = self._make_line(slope, intercept)
        model.fit_line(points_x, points_y, self.n_points,
                       self.list_slopes, self.list_intercepts, self.pos)
        self.assertAlmostEqual(self.list_slopes[self.pos], slope)
        self.assertAlmostEqual(self.list_intercepts[self.pos], intercept)


    def test_negative_slope(self):
        """
        slope = -1, intercept = 5, n_points = 10, clean data.
        fit_line should recover slope = -1, intercept = 5 exactly.
        """
        slope, intercept = -1, 5
        points_x, points_y = self._make_line(slope, intercept)
        model.fit_line(points_x, points_y, self.n_points,
                       self.list_slopes, self.list_intercepts, self.pos)
        self.assertAlmostEqual(self.list_slopes[self.pos], slope)
        self.assertAlmostEqual(self.list_intercepts[self.pos], intercept)


    def test_zero_slope(self):
        """
        slope = 0, intercept = 3, n_points = 10, clean data.
        fit_line should recover slope = 0, intercept = 3 exactly.
        """
        slope, intercept = 0, 3
        points_x, points_y = self._make_line(slope, intercept)
        model.fit_line(points_x, points_y, self.n_points,
                       self.list_slopes, self.list_intercepts, self.pos)
        self.assertAlmostEqual(self.list_slopes[self.pos], slope)
        self.assertAlmostEqual(self.list_intercepts[self.pos], intercept)


    def test_fractional_slope(self):
        """
        slope = 0.5, intercept = 0, n_points = 10, clean data.
        fit_line should recover slope = 0.5, intercept = 0 exactly.
        """
        slope, intercept = 0.5, 0
        points_x, points_y = self._make_line(slope, intercept)
        model.fit_line(points_x, points_y, self.n_points,
                       self.list_slopes, self.list_intercepts, self.pos)
        self.assertAlmostEqual(self.list_slopes[self.pos], slope)
        self.assertAlmostEqual(self.list_intercepts[self.pos], intercept)


    def test_minimum_points(self):
        """
        slope = 1, intercept = 0, n_points = 2, clean data.
        fit_line should recover slope = 1, intercept = 0 exactly
        at minimum sample size.
        """
        slope, intercept = 1, 0
        points_x, points_y = self._make_line(slope, intercept, n_points=2)
        model.fit_line(points_x, points_y, 2,
                       self.list_slopes, self.list_intercepts, self.pos)
        self.assertAlmostEqual(self.list_slopes[self.pos], slope)
        self.assertAlmostEqual(self.list_intercepts[self.pos], intercept)


    def test_pos_negative(self):
        """
        pos < 0, should return -1.
        """
        slope, intercept = 1, 0
        points_x, points_y = self._make_line(slope, intercept)
        ret = model.fit_line(points_x, points_y, self.n_points,
                             self.list_slopes, self.list_intercepts, -1)
        self.assertEqual(ret, -1)


    def test_n_points_less_than_2(self):
        """
        n_points = 1, should return -1.
        """
        slope, intercept = 1, 0
        points_x, points_y = self._make_line(slope, intercept)
        ret = model.fit_line(points_x, points_y, 1,
                             self.list_slopes, self.list_intercepts, self.pos)
        self.assertEqual(ret, -1)


    def test_all_x_equal(self):
        """
        all x values equal, vertical line, slope undefined.
        should return -1.
        """
        points_x = [1.0] * self.n_points
        points_y = [float(i) for i in range(self.n_points)]
        ret = model.fit_line(points_x, points_y, self.n_points,
                             self.list_slopes, self.list_intercepts, self.pos)
        self.assertEqual(ret, -1)


class TestPointsToLineDistances(unittest.TestCase):
    """
    Tests for points_to_line_distances which computes perpendicular distance
    from each point to a line defined by slope and intercept, storing results
    in distances list in place.

    The geometric distance formula used is:
        distances[i] = |slope * points_x[i] - points_y[i] + intercept|
                       / sqrt(1 + slope^2)

    Happy paths:
        all points on the line      all distances = 0.0
        point 1 unit above line     distance = 1 / sqrt(1 + slope^2)
        slope = 0, point above line distance = vertical distance
        negative slope              distances still positive (absolute value)

    Edge cases:
        n_points < 1                return -1
    """


if __name__ == '__main__':
    unittest.main()
