"""Pure checks of the predeclared C1 schedule, variance top-up and decision rule."""
import unittest

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from analysis import seed_summary
from c1_experiment import decision, paired_power, require_pilot, schedule, topup_rule


class C1Rules(unittest.TestCase):
    def test_matched_first_twenty_then_primary_only_topup(self):
        pilot = schedule("pilot")
        self.assertEqual(len(pilot), 120)
        self.assertEqual([s for _, s, _ in pilot], sorted(s for _, s, _ in pilot))
        self.assertEqual({s for a, s, _ in pilot if a == "es20"}, set(range(1, 21)))
        require_pilot(set(pilot))
        with self.assertRaisesRegex(ValueError, "Finish all"):
            require_pilot(set(pilot[:-1]))
        topup = schedule("topup", 63)
        self.assertEqual(len(topup), 129)
        self.assertEqual({a for a, _, _ in topup}, {"fixed250"})
        self.assertEqual({s for _, s, _ in topup}, set(range(21, 64)))

    def test_power_is_variance_only_and_budget_is_not_silently_capped(self):
        differences = np.linspace(-1, 1, 20)
        differences *= 2.7697508913 / differences.std(ddof=1)
        result = topup_rule(differences)
        self.assertEqual(result["target_pairs"], 63)
        self.assertEqual(topup_rule(differences + 20)["target_pairs"], 63)
        self.assertGreaterEqual(paired_power(63, result["sd_pp"]), .8)
        self.assertLess(paired_power(62, result["sd_pp"]), .8)
        self.assertEqual(topup_rule(np.zeros(20))["target_pairs"], 63)
        self.assertIsNone(topup_rule(differences * 10)["target_pairs"])
        for invalid in [differences[:-1], np.full(20, np.nan)]:
            with self.assertRaises(ValueError):
                topup_rule(invalid)

    def test_decision_uses_zero_and_one_pp_without_equivalence(self):
        self.assertIn("incomplete", decision(2, 1.1, 2.9, False))
        self.assertIn("at least 1 pp", decision(2, 1, 3, True))
        self.assertIn("not established", decision(.5, .1, .9, True))
        self.assertIn("favors basic", decision(-2, -3, -1, True))
        self.assertIn("no equivalence", decision(0, -.5, .5, True))
        self.assertIn("inconclusive", decision(.5, 0, 1.5, True))

    def test_paired_interval_matches_scipy_and_missing_pairs_are_not_imputed(self):
        d = np.array([0., 1., 2., 4., -1.])
        runs = pd.DataFrame([dict(seed=i, variant=v, acc=90 + (x if v == "gan" else 0))
                             for i, x in enumerate(d, 1) for v in ["basic", "gan"]])
        result = seed_summary(runs).iloc[0]
        expected = ttest_1samp(d, 0).confidence_interval()
        self.assertAlmostEqual(result.ci95_lo, expected.low)
        self.assertAlmostEqual(result.ci95_hi, expected.high)
        self.assertEqual(seed_summary(runs.iloc[:-1]).iloc[0].n_seeds, 4)
        zero = seed_summary(runs.assign(acc=90)).iloc[0]
        self.assertEqual(zero.wilcoxon_p, 1)
        self.assertEqual(zero.ci95_lo, 0)
        self.assertEqual(zero.zero_differences, 5)


if __name__ == "__main__":
    unittest.main()
