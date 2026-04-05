import evaluate

import math
import unittest


class TestModelError(unittest.TestCase):
    """
    Tests for model_error which measures the Euclidean distance between the
    estimated model parameters and the true model parameters as:
        sqrt((slope - true_slope)^2 + (intercept - true_intercept)^2)

    Happy paths:
        exact recovery, error should be 0.0
        slope differs only, error should equal abs(slope - true_slope)
        intercept differs only, error should equal abs(intercept - true_intercept)
        both differ, error should equal sqrt(ds^2 + di^2)
        negative differences, error should still be positive
    """

    def setUp(self):
        self.true_slope = 2.0
        self.true_intercept = 5.0


    def test_exact_recovery(self):
        """
        Estimated model equals true model. Error should be exactly 0.0.
        """
        error = evaluate.model_error(self.true_slope, self.true_intercept,
                                     self.true_slope, self.true_intercept)
        self.assertAlmostEqual(error, 0.0)


    def test_slope_differs(self):
        """
        Slope differs by 1.0, intercept matches.
        Error should equal abs(slope - true_slope) = 1.0.
        """
        slope = self.true_slope + 1.0
        error = evaluate.model_error(slope, self.true_intercept,
                                     self.true_slope, self.true_intercept)
        self.assertAlmostEqual(error, 1.0)


    def test_intercept_differs(self):
        """
        Intercept differs by 2.0, slope matches.
        Error should equal abs(intercept - true_intercept) = 2.0.
        """
        intercept = self.true_intercept + 2.0
        error = evaluate.model_error(self.true_slope, intercept,
                                     self.true_slope, self.true_intercept)
        self.assertAlmostEqual(error, 2.0)


    def test_both_differ(self):
        """
        Slope differs by 3.0, intercept differs by 4.0.
        Error should equal sqrt(3^2 + 4^2) = 5.0.
        """
        slope = self.true_slope + 3.0
        intercept = self.true_intercept + 4.0
        error = evaluate.model_error(slope, intercept,
                                     self.true_slope, self.true_intercept)
        self.assertAlmostEqual(error, 5.0)


    def test_negative_differences(self):
        """
        Estimated slope and intercept are less than true values.
        Error should still be positive.
        """
        slope = self.true_slope - 3.0
        intercept = self.true_intercept - 4.0
        error = evaluate.model_error(slope, intercept,
                                     self.true_slope, self.true_intercept)
        self.assertGreater(error, 0.0)  # positive
        self.assertAlmostEqual(error, 5.0)


if __name__ == "__main__":
    unittest.main()

