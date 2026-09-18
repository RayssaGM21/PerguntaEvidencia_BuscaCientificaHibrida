from __future__ import annotations

import unittest

from src.evaluation.significance import paired_bootstrap


class SignificanceTest(unittest.TestCase):
    def test_paired_bootstrap_positive_difference(self) -> None:
        result = paired_bootstrap([0.1, 0.2, 0.3], [0.2, 0.3, 0.4], samples=100, seed=1)

        self.assertAlmostEqual(result.mean_difference, 0.1)
        self.assertLessEqual(result.ci_lower, result.mean_difference)
        self.assertGreaterEqual(result.ci_upper, result.mean_difference)


if __name__ == "__main__":
    unittest.main()
