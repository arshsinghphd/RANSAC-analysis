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
        self.n = 10
        self.x_min = 0
        self.x_max = self.n - 1
        self.pos = 3
        self.list_slopes = [float("inf")] * (self.pos + 1)
        self.list_intercepts = [float("inf")] * (self.pos + 1)


    def _make_line(self, points_x, points_y, slope, intercept, n=None):
        n = n if n else self.n
        generator.make_inliers(points_x, points_y, n, slope, intercept,
                               self.x_min, self.x_max)


    def _assert_fit(self, slope, intercept):
        self.assertAlmostEqual(self.list_slopes[self.pos], slope)
        self.assertAlmostEqual(self.list_intercepts[self.pos], intercept)


    def test_unit_slope_zero_intercept(self):
        """
        slope = 1, intercept = 0, n_points = 10, clean data.
        fit_line should recover slope = 1, intercept = 0 exactly.
        """
        points_x = [float("inf")] * self.n
        points_y = [float("inf")] * self.n
        self._make_line(points_x, points_y, 1, 0)
        model.fit_line(points_x, points_y, self.n,
                       self.list_slopes, self.list_intercepts, self.pos)
        self._assert_fit(1, 0)


    def test_negative_slope(self):
        """
        slope = -1, intercept = 5, n_points = 10, clean data.
        fit_line should recover slope = -1, intercept = 5 exactly.
        """
        points_x = [float("inf")] * self.n
        points_y = [float("inf")] * self.n
        self._make_line(points_x, points_y, -1, 5)
        model.fit_line(points_x, points_y, self.n,
                       self.list_slopes, self.list_intercepts, self.pos)
        self._assert_fit(-1, 5)


    def test_zero_slope(self):
        """
        slope = 0, intercept = 3, n_points = 10, clean data.
        fit_line should recover slope = 0, intercept = 3 exactly.
        """
        points_x = [float("inf")] * self.n
        points_y = [float("inf")] * self.n
        self._make_line(points_x, points_y, 0, 3)
        model.fit_line(points_x, points_y, self.n,
                       self.list_slopes, self.list_intercepts, self.pos)
        self._assert_fit(0, 3)


    def test_fractional_slope(self):
        """
        slope = 0.5, intercept = 0, n_points = 10, clean data.
        fit_line should recover slope = 0.5, intercept = 0 exactly.
        """
        points_x = [float("inf")] * self.n
        points_y = [float("inf")] * self.n
        self._make_line(points_x, points_y, 0.5, 0)
        model.fit_line(points_x, points_y, self.n,
                       self.list_slopes, self.list_intercepts, self.pos)
        self._assert_fit(0.5, 0)


    def test_minimum_points(self):
        """
        slope = 1, intercept = 0, n_points = 2, clean data.
        fit_line should recover slope = 1, intercept = 0 exactly
        at minimum sample size.
        """
        points_x = [float("inf")] * 2
        points_y = [float("inf")] * 2
        self._make_line(points_x, points_y, 1, 0, n=2)
        model.fit_line(points_x, points_y, 2,
                       self.list_slopes, self.list_intercepts, self.pos)
        self._assert_fit(1, 0)


    def test_pos_negative(self):
        """
        pos < 0, should return -1.
        """
        points_x = [float("inf")] * self.n
        points_y = [float("inf")] * self.n
        self._make_line(points_x, points_y, 1, 0)
        ret = model.fit_line(points_x, points_y, self.n,
                             self.list_slopes, self.list_intercepts, -1)
        self.assertEqual(ret, -1)


    def test_n_points_less_than_2(self):
        """
        n_points = 1, should return -1.
        """
        points_x = [float("inf")] * self.n
        points_y = [float("inf")] * self.n
        self._make_line(points_x, points_y, 1, 0)
        ret = model.fit_line(points_x, points_y, 1,
                             self.list_slopes, self.list_intercepts, self.pos)
        self.assertEqual(ret, -1)


    def test_all_x_equal(self):
        """
        all x values equal, vertical line, slope undefined.
        should return -1.
        """
        points_x = [1.0] * self.n
        points_y = [float(i) for i in range(self.n)]
        ret = model.fit_line(points_x, points_y, self.n,
                             self.list_slopes, self.list_intercepts, self.pos)
        self.assertEqual(ret, -1)


class TestPointsToLineDistances(unittest.TestCase):
    """
    Tests for points_to_line_distances which computes perpendicular distance
    from each point to a line defined by slope and intercept, storing results
    in distances list in place.

    The geometric distance formula used is:
        distances[i] = |slope * points_x[i] - points_y[i] + intercept|
                       / sqrt(1 + slope squared)

    Happy paths:
        all points on the line, all distances equal 0.0
        points 1 unit above the line, distance equals 1 / sqrt(1 + slope squared)
        slope = 0, points 2 units above line, distance equals 2.0
        slope = -1, points 1 unit above line, distances positive due to absolute value

    Edge cases:
        n_points = 0, should return -1
    """

    def setUp(self):
        self.n = 10
        self.x_min = 0
        self.x_max = self.n - 1
        self.distances = [float("inf")] * self.n


    def _make_line(self, points_x, points_y, slope, intercept, n=None):
        n = n if n else self.n
        generator.make_inliers(points_x, points_y, n, slope, intercept,
                               self.x_min, self.x_max)


    def _assert_distances(self, distances, expected, n=None):
        n = n if n else self.n
        for i in range(n):
            self.assertAlmostEqual(distances[i], expected)


    def test_points_on_line(self):
        """
        All points lie exactly on the line.
        All distances should equal 0.0.
        """
        slope, intercept = 1, 0
        points_x = [float("inf")] * self.n
        points_y = [float("inf")] * self.n
        self._make_line(points_x, points_y, slope, intercept)
        model.points_to_line_distances(points_x, points_y, self.n,
                                       slope, intercept, self.distances)
        self._assert_distances(self.distances, 0.0)


    def test_points_one_unit_above_line(self):
        """
        All points shifted 1 unit above the line.
        Each distance should equal 1 / sqrt(1 + slope squared).
        """
        slope, intercept = 1, 0
        points_x = [float("inf")] * self.n
        points_y = [float("inf")] * self.n
        self._make_line(points_x, points_y, slope, intercept)
        for i in range(self.n):
            points_y[i] += 1.0
        model.points_to_line_distances(points_x, points_y, self.n,
                                       slope, intercept, self.distances)
        expected = 1.0 / math.sqrt(1 + slope * slope)
        self._assert_distances(self.distances, expected)


    def test_zero_slope_points_above_line(self):
        """
        slope = 0, intercept = 0. Points shifted 2 units above.
        Each distance should equal 2.0.
        """
        slope, intercept = 0, 0
        points_x = [float("inf")] * self.n
        points_y = [float("inf")] * self.n
        self._make_line(points_x, points_y, slope, intercept)
        for i in range(self.n):
            points_y[i] += 2.0
        model.points_to_line_distances(points_x, points_y, self.n,
                                       slope, intercept, self.distances)
        self._assert_distances(self.distances, 2.0)


    def test_negative_slope_points_above_line(self):
        """
        slope = -1, intercept = 0. Points shifted 1 unit above.
        Distances should be positive due to absolute value in formula.
        """
        slope, intercept = -1, 0
        points_x = [float("inf")] * self.n
        points_y = [float("inf")] * self.n
        self._make_line(points_x, points_y, slope, intercept)
        for i in range(self.n):
            points_y[i] += 1.0
        model.points_to_line_distances(points_x, points_y, self.n,
                                       slope, intercept, self.distances)
        expected = 1.0 / math.sqrt(1 + slope * slope)
        self._assert_distances(self.distances, expected)


    def test_n_points_less_than_1(self):
        """
        n_points = 0, should return -1.
        """
        points_x = []
        points_y = []
        distances = []
        n_points = 0
        slope, intercept = 1, 0
        ret = model.points_to_line_distances(points_x, points_y, n_points,
                                             slope, intercept, distances)
        self.assertEqual(ret, -1)


if __name__ == '__main__':
    unittest.main()
