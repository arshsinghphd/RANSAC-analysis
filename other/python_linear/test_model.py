"""
Tests for model.py functions: fit_line and points_to_line_distances.

fit_line estimates slope and intercept from n_points using least squares,
storing results at position pos in list_slopes and list_intercepts.

points_to_line_distances computes perpendicular distance from each point to a
line defined by slope and intercept, storing results in distances.

Run from terminal as follows:
    $ python -m unittest test_model.py
"""

import generator
import model
import ransac

import math
import unittest


class TestFindInliers(unittest.TestCase):
    """
    Tests for find_inliers which collects inliers from points by computing
    perpendicular distance from each point to a line. Points within threshold
    distance are appended to inliers_x and inliers_y in place.

    Happy paths:
        all points on the line, all points collected as inliers
        no points within threshold, inliers_x and inliers_y remain empty
        mixed points, only those within threshold collected

    Edge cases:
        n_points < 1,       should return -1
        threshold <= 0,     should return -1
    """

    def setUp(self):
        self.n = 10
        self.x_min = 0
        self.x_max = self.n - 1


    def test_all_points_on_line(self):
        """
        All points lie exactly on the line.
        All points should be collected as inliers.
        """
        slope = 1
        intercept = 0
        threshold = 0.1
        points_x = [i for i in range(self.n)]
        points_y = [i for i in range(self.n)]
        # initiate inliers
        inliers_x = []
        inliers_y = []
        # find inliers
        model.find_inliers(points_x, points_y, self.n, slope, intercept,
                           threshold, inliers_x, inliers_y)
        n_inliers = len(inliers_x)
        # assertions
        self.assertEqual(self.n, n_inliers)
        for i in range(self.n):
            self.assertAlmostEqual(points_x[i], inliers_x[i])
            self.assertAlmostEqual(points_y[i], inliers_y[i])


    def test_no_points_within_threshold(self):
        """
        All points far from the line.
        inliers_x and inliers_y should remain empty.
        """
        slope = 1
        intercept = 0
        threshold = 0.1
        points_x = [i for i in range(self.n)]
        points_y = [i + 10 * threshold for i in range(self.n)]
        # initiate inliers
        inliers_x = []
        inliers_y = []
        # find inliers
        model.find_inliers(points_x, points_y, self.n, slope, intercept,
                           threshold, inliers_x, inliers_y)
        n_inliers = len(inliers_x)
        # assertions
        self.assertEqual(n_inliers, 0)


    def test_mixed_points(self):
        """
        Half the points on the line, half far from the line.
        Only points within threshold should be collected.
        """
        slope = 1
        intercept = 0
        threshold = 0.1
        points_x = [i for i in range(self.n)]
        points_y = [i for i in range(self.n)]
        for i in range(self.n // 2):
            points_y[i] = points_y[i] + 10 * threshold
        # initiate inliers
        inliers_x = []
        inliers_y = []
        # find inliers
        model.find_inliers(points_x, points_y, self.n, slope, intercept,
                           threshold, inliers_x, inliers_y)
        n_inliers = len(inliers_x)
        # assertions
        self.assertEqual(n_inliers, self.n - self.n // 2)


    def test_n_points_less_than_1(self):
        """
        n_points = 0, should return -1.
        """
        slope = 1
        intercept = 0
        threshold = 0.1
        points_x = [i for i in range(self.n)]
        points_y = [i for i in range(self.n)]
        # initiate inliers
        inliers_x = []
        inliers_y = []
        # find inliers, overwrite n = 0
        ret = model.find_inliers(points_x, points_y, 0, slope, intercept,
                           threshold, inliers_x, inliers_y)
        # assertions
        self.assertEqual(ret, -1)


    def test_threshold_zero(self):
        """
        threshold = 0, should return -1.
        """
        slope = 1
        intercept = 0
        threshold = 0       # threshold = 0, testing
        points_x = [i for i in range(self.n)]
        points_y = [i for i in range(self.n)]
        # initiate inliers
        inliers_x = []
        inliers_y = []
        # find inliers
        ret = model.find_inliers(points_x, points_y, self.n, slope, intercept,
                           threshold, inliers_x, inliers_y)
        # assertions
        self.assertEqual(ret, -1)


    def test_threshold_negative(self):
        """
        threshold = -1, should return -1.
        """
        slope = 1
        intercept = 0
        threshold = -1       # threshold < 0, testing
        points_x = [i for i in range(self.n)]
        points_y = [i for i in range(self.n)]
        # initiate inliers
        inliers_x = []
        inliers_y = []
        # find inliers
        ret = model.find_inliers(points_x, points_y, self.n, slope, intercept,
                           threshold, inliers_x, inliers_y)
        # assertions
        self.assertEqual(ret, -1)


class TestStitchModels(unittest.TestCase):
    """
    Tests for stitch_models which combines inliers from two overlapping graphs
    into a single refined linear model. Collects inliers from each graph using
    their respective RANSAC-recovered models, combines them, and refits one
    line using least squares.

    Happy paths:
        two clean overlapping graphs with same true model
            combined fit should recover true slope and intercept exactly
        two noisy overlapping graphs with same true model
            combined fit should be closer to true model than either
            individual fit

    Edge cases:
        n1 < 2,             should return -1
        n2 < 2,             should return -1
        threshold <= 0,     should return -1
        pos < 0,            should return -1
        no inliers graph 1, should return -1
        no inliers graph 2, should return -1
    """

    def setUp(self):
        self.n = 100
        self.x_min = 0
        self.x_max = self.n - 1
        self.true_slope = 2.0
        self.true_intercept = 5.0
        self.pos = 0
        self.list_slopes = [float("inf")]
        self.list_intercepts = [float("inf")]


    def test_clean_overlapping_graphs(self):
        """
        Two clean overlapping graphs built from the same true model.
        Combined fit should recover true slope and intercept exactly.
        """
        # graph 1
        x1_min = self.x_min
        x1_max = 60
        n_points1 = 60
        points_x1 = [float("inf")] * x1_max
        points_y1 = [float("inf")] * x1_max
        generator.make_inliers(points_x1, points_y1,
                               n_points1,
                               self.true_slope, self.true_intercept,
                               x1_min, x1_max - 1)
        slope1 = self.true_slope            # since no noise or outliers
        intercept1 = self.true_intercept    # since no noise or outliers
        # graph 2
        x2_min = 40
        x2_max = self.x_max
        n_points2 = x2_max - x1_max
        points_x2 = [float("inf")] * n_points2
        points_y2 = [float("inf")] * n_points2
        generator.make_inliers(points_x2, points_y2,
                               n_points2,
                               self.true_slope, self.true_intercept,
                               x2_min, x2_max - 1)
        slope2 = self.true_slope            # since no noise or outliers
        intercept2 = self.true_intercept    # since no noise or outliers
        # stitch graphs
        list_slopes = self.list_slopes
        list_intercepts = self.list_intercepts
        pos = self.pos
        threshold = 2
        model.stitch_models(points_x1, points_y1, n_points1,
                  slope1, intercept1,
                  points_x2, points_y2, n_points2,
                  slope2, intercept2,
                  threshold,
                  list_slopes, list_intercepts, pos)
        slope = list_slopes[pos]
        intercept = list_intercepts[pos]
        # assertions
        self.assertEqual(slope, self.true_slope)
        self.assertEqual(intercept, self.true_intercept)


    def test_noisy_overlapping_graphs(self):
        """
        Two noisy overlapping graphs built from the same true model.
        Combined fit should recover true slope and intercept within tolerance.
        """
        std = 2
        # create graph 1
        x1_min = self.x_min
        x1_max = 60
        n_inliers1 = 60
        points_x1 = [float("inf")] * n_inliers1
        points_y1 = [float("inf")] * n_inliers1
        generator.make_inliers(points_x1, points_y1,
                               n_inliers1,
                               self.true_slope, self.true_intercept,
                               x1_min, x1_max - 1)
        # make it noisy
        generator.add_gaussian_noise(points_y1, n_inliers1, std)
        # add outliers
        epsilon1 = 0.1
        n_outliers1 = int(epsilon1 * n_inliers1)
        generator.add_outliers(points_x1, points_y1, n_inliers1, n_outliers1,
                 self.true_slope, self.true_intercept, std)
        # find solution using RANSAC
        n_points1 = n_inliers1 + n_outliers1
        threshold1 = ransac.compute_t(points_x1, points_y1, n_points1)
        n_params = 2
        return_array1 = [float("inf")] * 9 
        k_resample1 = ransac.compute_k(epsilon1, n_params, failure_prob=0.01)
        ransac.ransac(points_x1, points_y1, n_points1, n_params, k_resample1,
                      threshold1, n_inliers1, return_array1)
        
        # make graph 2
        x2_min = 40
        x2_max = self.x_max
        n_inliers2 = x2_max - x1_max + 1
        points_x2 = [float("inf")] * n_inliers2
        points_y2 = [float("inf")] * n_inliers2
        generator.make_inliers(points_x2, points_y2,
                               n_inliers2,
                               self.true_slope, self.true_intercept,
                               x2_min, x2_max - 1)
        # make it noisy
        generator.add_gaussian_noise(points_y2, n_inliers2, std)
        # add outliers
        epsilon2 = 0.1
        n_outliers2 = int(epsilon2 * n_inliers2)
        generator.add_outliers(points_x2, points_y2, n_inliers2, n_outliers2,
                               self.true_slope, self.true_intercept, std)
        # find solution using RANSAC
        n_points2 = n_inliers2 + n_outliers2
        threshold2 = ransac.compute_t(points_x2, points_y2, n_points2)
        n_params = 2
        k_resample2 = ransac.compute_k(epsilon2, n_params, failure_prob=0.01)
        return_array2 = [float("inf")] * 9 
        ransac.ransac(points_x2, points_y2, n_points2, n_params, k_resample2,
                      threshold2, n_inliers2, return_array2)
        # stitch graphs
        list_slopes = self.list_slopes
        list_intercepts = self.list_intercepts
        pos = self.pos
        threshold = max(threshold1, threshold2)
        model.stitch_models(points_x1, points_y1, n_points1,
                  return_array1[5], return_array1[6],
                  points_x2, points_y2, n_points2,
                  return_array2[5], return_array2[6],
                  threshold,
                  list_slopes, list_intercepts, pos)
        slope = list_slopes[pos]
        intercept = list_intercepts[pos]
        # assertions
        error = model.model_error(slope, intercept,
                          self.true_slope, self.true_intercept)
        self.assertLess(error, threshold)


    def test_n1_less_than_2(self):
        """
        n1 = 1, should return -1.
        """
        std = 1
        # graph 1
        x1_min = self.x_min
        x1_max = 60
        n_points1 = 60
        points_x1 = [float("inf")] * n_points1
        points_y1 = [float("inf")] * n_points1
        generator.make_inliers(points_x1, points_y1,
                               n_points1,
                               self.true_slope, self.true_intercept,
                               x1_min, x1_max - 1)
        generator.add_gaussian_noise(points_y1, n_points1, std)
        list_slopes1 = self.list_slopes
        list_intercepts1 = self.list_intercepts
        pos = self.pos
        model.fit_line(points_x1, points_y1, n_points1,
                       list_slopes1, list_intercepts1, pos)
        slope1 = list_slopes1[pos]
        intercept1 = list_intercepts1[pos]
        # graph 2
        x2_min = 40
        x2_max = self.x_max
        n_points2 = x2_max - x1_max
        points_x2 = [float("inf")] * n_points2
        points_y2 = [float("inf")] * n_points2
        generator.make_inliers(points_x2, points_y2,
                               n_points2,
                               self.true_slope, self.true_intercept,
                               x2_min, x2_max - 1)
        generator.add_gaussian_noise(points_y2, n_points2, std)
        list_slopes2 = self.list_slopes
        list_intercepts2 = self.list_intercepts
        pos = self.pos
        model.fit_line(points_x2, points_y2, n_points2,
                       list_slopes2, list_intercepts2, pos)
        slope2 = list_slopes2[pos]
        intercept2 = list_intercepts2[pos]
        # stitch graphs
        list_slopes = self.list_slopes
        list_intercepts = self.list_intercepts
        pos = self.pos
        threshold = 2 * std
        n_points1 = 1           # overwrite
        ret = model.stitch_models(points_x1, points_y1, n_points1,
                  slope1, intercept1,
                  points_x2, points_y2, n_points2,
                  slope2, intercept2,
                  threshold,
                  list_slopes, list_intercepts, pos)
        self.assertEqual(ret, -1)


    def test_n2_less_than_2(self):
        """
        n2 = 1, should return -1.
        """
        std = 1
        # graph 1
        x1_min = self.x_min
        x1_max = 60
        n_points1 = 60
        points_x1 = [float("inf")] * n_points1
        points_y1 = [float("inf")] * n_points1
        generator.make_inliers(points_x1, points_y1,
                               n_points1,
                               self.true_slope, self.true_intercept,
                               x1_min, x1_max - 1)
        generator.add_gaussian_noise(points_y1, n_points1, std)
        list_slopes1 = self.list_slopes
        list_intercepts1 = self.list_intercepts
        pos = self.pos
        model.fit_line(points_x1, points_y1, n_points1,
                       list_slopes1, list_intercepts1, pos)
        slope1 = list_slopes1[pos]
        intercept1 = list_intercepts1[pos]
        # graph 2
        x2_min = 40
        x2_max = self.x_max
        n_points2 = x2_max - x1_max
        points_x2 = [float("inf")] * n_points2
        points_y2 = [float("inf")] * n_points2
        generator.make_inliers(points_x2, points_y2,
                               n_points2,
                               self.true_slope, self.true_intercept,
                               x2_min, x2_max - 1)
        generator.add_gaussian_noise(points_y2, n_points2, std)
        list_slopes2 = self.list_slopes
        list_intercepts2 = self.list_intercepts
        pos = self.pos
        model.fit_line(points_x2, points_y2, n_points2,
                       list_slopes2, list_intercepts2, pos)
        slope2 = list_slopes2[pos]
        intercept2 = list_intercepts2[pos]
        # stitch graphs
        list_slopes = self.list_slopes
        list_intercepts = self.list_intercepts
        pos = self.pos
        threshold = 2 * std
        n_points2 = 1           # overwrite
        ret = model.stitch_models(points_x1, points_y1, n_points1,
                  slope1, intercept1,
                  points_x2, points_y2, n_points2,
                  slope2, intercept2,
                  threshold,
                  list_slopes, list_intercepts, pos)
        self.assertEqual(ret, -1)


    def test_threshold_zero(self):
        """
        threshold = 0, should return -1.
        """
        std = 1
        # graph 1
        x1_min = self.x_min
        x1_max = 60
        n_points1 = 60
        points_x1 = [float("inf")] * n_points1
        points_y1 = [float("inf")] * n_points1
        generator.make_inliers(points_x1, points_y1,
                               n_points1,
                               self.true_slope, self.true_intercept,
                               x1_min, x1_max - 1)
        generator.add_gaussian_noise(points_y1, n_points1, std)
        list_slopes1 = self.list_slopes
        list_intercepts1 = self.list_intercepts
        pos = self.pos
        model.fit_line(points_x1, points_y1, n_points1,
                       list_slopes1, list_intercepts1, pos)
        slope1 = list_slopes1[pos]
        intercept1 = list_intercepts1[pos]
        # graph 2
        x2_min = 40
        x2_max = self.x_max
        n_points2 = x2_max - x1_max
        points_x2 = [float("inf")] * n_points2
        points_y2 = [float("inf")] * n_points2
        generator.make_inliers(points_x2, points_y2,
                               n_points2,
                               self.true_slope, self.true_intercept,
                               x2_min, x2_max - 1)
        generator.add_gaussian_noise(points_y2, n_points2, std)
        list_slopes2 = self.list_slopes
        list_intercepts2 = self.list_intercepts
        pos = self.pos
        model.fit_line(points_x2, points_y2, n_points2,
                       list_slopes2, list_intercepts2, pos)
        slope2 = list_slopes2[pos]
        intercept2 = list_intercepts2[pos]
        # stitch graphs
        list_slopes = self.list_slopes
        list_intercepts = self.list_intercepts
        pos = self.pos
        threshold = 0       # overwrite to test
        ret = model.stitch_models(points_x1, points_y1, n_points1,
                  slope1, intercept1,
                  points_x2, points_y2, n_points2,
                  slope2, intercept2,
                  threshold,
                  list_slopes, list_intercepts, pos)
        self.assertEqual(ret, -1)


    def test_threshold_negative(self):
        """
        threshold = -1, should return -1.
        """
        std = 1
        # graph 1
        x1_min = self.x_min
        x1_max = 60
        n_points1 = 60
        points_x1 = [float("inf")] * n_points1
        points_y1 = [float("inf")] * n_points1
        generator.make_inliers(points_x1, points_y1,
                               n_points1,
                               self.true_slope, self.true_intercept,
                               x1_min, x1_max - 1)
        generator.add_gaussian_noise(points_y1, n_points1, std)
        list_slopes1 = self.list_slopes
        list_intercepts1 = self.list_intercepts
        pos = self.pos
        model.fit_line(points_x1, points_y1, n_points1,
                       list_slopes1, list_intercepts1, pos)
        slope1 = list_slopes1[pos]
        intercept1 = list_intercepts1[pos]
        # graph 2
        x2_min = 40
        x2_max = self.x_max
        n_points2 = x2_max - x1_max
        points_x2 = [float("inf")] * n_points2
        points_y2 = [float("inf")] * n_points2
        generator.make_inliers(points_x2, points_y2,
                               n_points2,
                               self.true_slope, self.true_intercept,
                               x2_min, x2_max - 1)
        generator.add_gaussian_noise(points_y2, n_points2, std)
        list_slopes2 = self.list_slopes
        list_intercepts2 = self.list_intercepts
        pos = self.pos
        model.fit_line(points_x2, points_y2, n_points2,
                       list_slopes2, list_intercepts2, pos)
        slope2 = list_slopes2[pos]
        intercept2 = list_intercepts2[pos]
        # stitch graphs
        list_slopes = self.list_slopes
        list_intercepts = self.list_intercepts
        pos = self.pos
        threshold = -1       # overwrite to test
        ret = model.stitch_models(points_x1, points_y1, n_points1,
                  slope1, intercept1,
                  points_x2, points_y2, n_points2,
                  slope2, intercept2,
                  threshold,
                  list_slopes, list_intercepts, pos)
        self.assertEqual(ret, -1)


    def test_pos_negative(self):
        """
        pos = -1, should return -1.
        """
        std = 1
        # graph 1
        x1_min = self.x_min
        x1_max = 60
        n_points1 = 60
        points_x1 = [float("inf")] * n_points1
        points_y1 = [float("inf")] * n_points1
        generator.make_inliers(points_x1, points_y1,
                               n_points1,
                               self.true_slope, self.true_intercept,
                               x1_min, x1_max - 1)
        generator.add_gaussian_noise(points_y1, n_points1, std)
        list_slopes1 = self.list_slopes
        list_intercepts1 = self.list_intercepts
        pos = self.pos
        model.fit_line(points_x1, points_y1, n_points1,
                       list_slopes1, list_intercepts1, pos)
        slope1 = list_slopes1[pos]
        intercept1 = list_intercepts1[pos]
        # graph 2
        x2_min = 40
        x2_max = self.x_max
        n_points2 = x2_max - x1_max
        points_x2 = [float("inf")] * n_points2
        points_y2 = [float("inf")] * n_points2
        generator.make_inliers(points_x2, points_y2,
                               n_points2,
                               self.true_slope, self.true_intercept,
                               x2_min, x2_max - 1)
        generator.add_gaussian_noise(points_y2, n_points2, std)
        list_slopes2 = self.list_slopes
        list_intercepts2 = self.list_intercepts
        pos = self.pos
        model.fit_line(points_x2, points_y2, n_points2,
                       list_slopes2, list_intercepts2, pos)
        slope2 = list_slopes2[pos]
        intercept2 = list_intercepts2[pos]
        # stitch graphs
        list_slopes = self.list_slopes
        list_intercepts = self.list_intercepts
        pos = -1        # overwrite to test
        threshold = 2 * std
        ret = model.stitch_models(points_x1, points_y1, n_points1,
                  slope1, intercept1,
                  points_x2, points_y2, n_points2,
                  slope2, intercept2,
                  threshold,
                  list_slopes, list_intercepts, pos)
        self.assertEqual(ret, -1)


    def test_no_inliers_graph1(self):
        """
        Threshold too small for graph 1 to find any inliers.
        Should return -1.
        """
        std = 1
        # graph 1
        x1_min = self.x_min
        x1_max = 60
        n_points1 = 60
        points_x1 = [float("inf")] * n_points1
        points_y1 = [float("inf")] * n_points1
        generator.make_inliers(points_x1, points_y1,
                               n_points1,
                               self.true_slope, self.true_intercept,
                               x1_min, x1_max - 1)
        generator.add_gaussian_noise(points_y1, n_points1, std)
        list_slopes1 = self.list_slopes
        list_intercepts1 = self.list_intercepts
        pos = self.pos
        model.fit_line(points_x1, points_y1, n_points1,
                       list_slopes1, list_intercepts1, pos)
        slope1 = list_slopes1[pos]
        intercept1 = list_intercepts1[pos]
        # graph 2
        x2_min = 40
        x2_max = self.x_max
        n_points2 = x2_max - x1_max
        points_x2 = [float("inf")] * n_points2
        points_y2 = [float("inf")] * n_points2
        generator.make_inliers(points_x2, points_y2,
                               n_points2,
                               self.true_slope, self.true_intercept,
                               x2_min, x2_max - 1)
        slope2 = self.true_slope
        intercept2 = self.true_intercept
        # stitch graphs
        list_slopes = self.list_slopes
        list_intercepts = self.list_intercepts
        pos = self.pos
        threshold = 1e-6        # overwrite to test,
                                # stitch_models will find 0 inliers
        ret = model.stitch_models(points_x1, points_y1, n_points1,
                  slope1, intercept1,
                  points_x2, points_y2, n_points2,
                  slope2, intercept2,
                  threshold,
                  list_slopes, list_intercepts, pos)
        self.assertEqual(ret, -1)


    def test_no_inliers_graph2(self):
        """
        Threshold too small for graph 2 to find any inliers.
        Should return -1.
        """
        std = 1
        # graph 1
        x1_min = self.x_min
        x1_max = 60
        n_points1 = 60
        points_x1 = [float("inf")] * n_points1
        points_y1 = [float("inf")] * n_points1
        generator.make_inliers(points_x1, points_y1,
                               n_points1,
                               self.true_slope, self.true_intercept,
                               x1_min, x1_max - 1)
        slope1 = self.true_slope
        intercept1 = self.true_intercept
        # graph 2
        x2_min = 40
        x2_max = self.x_max
        n_points2 = x2_max - x1_max
        points_x2 = [float("inf")] * n_points2
        points_y2 = [float("inf")] * n_points2
        generator.make_inliers(points_x2, points_y2,
                               n_points2,
                               self.true_slope, self.true_intercept,
                               x2_min, x2_max - 1)
        generator.add_gaussian_noise(points_y2, n_points2, std)
        list_slopes2 = self.list_slopes
        list_intercepts2 = self.list_intercepts
        pos = self.pos
        model.fit_line(points_x2, points_y2, n_points2,
                       list_slopes2, list_intercepts2, pos)
        slope2 = list_slopes2[pos]
        intercept2 = list_intercepts2[pos]
        # stitch graphs
        list_slopes = self.list_slopes
        list_intercepts = self.list_intercepts
        pos = self.pos
        threshold = 1e-6        # overwrite to test,
                                # stitch_models will find 0 inliers
        ret = model.stitch_models(points_x1, points_y1, n_points1,
                  slope1, intercept1,
                  points_x2, points_y2, n_points2,
                  slope2, intercept2,
                  threshold,
                  list_slopes, list_intercepts, pos)
        self.assertEqual(ret, -1)


class TestFitLine(unittest.TestCase):
    """
    Tests for fit_line which estimates slope and intercept from n_points using
    least squares, storing results at position pos in list_slopes and
    list_intercepts.

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
        fit_line should recover slope = 1, intercept = 0 exactly at minimum
        sample size.
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
        points 1 unit above the line,
            distance = 1 / sqrt(1 + slope squared)
        slope = 0, points 2 units above line, distance equals 2.0
        slope = -1, points 1 unit above line, distances positive due to
        absolute value

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
