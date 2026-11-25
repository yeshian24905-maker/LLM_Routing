"""Checks for the prior-centred primal-dual Thompson policy and the runner options it needs."""

import math
import statistics
import unittest

from bandit.prior_thompson import PriorPrimalDualThompson


ARMS = ("cheap", "strong")
COSTS = {"cheap": 1.0, "strong": 4.0}
PRIORS = {"easy": {"cheap": 0.95, "strong": 0.96}, "hard": {"cheap": 0.3, "strong": 0.9}}


class PriorThompsonTests(unittest.TestCase):
    def test_infinite_kappa_uses_predictions_as_they_are(self):
        policy = PriorPrimalDualThompson(ARMS, COSTS, 0.8, PRIORS, kappa=math.inf, lambda0=10)
        self.assertEqual(policy.sample("g", "cheap", "hard"), 0.3)
        self.assertEqual(policy.select("g", "easy"), "cheap")
        self.assertEqual(policy.select("g", "hard"), "strong")

    def test_group_offset_corrects_a_biased_prediction(self):
        policy = PriorPrimalDualThompson(ARMS, COSTS, 0.8, PRIORS, kappa=10)
        for _ in range(30):
            policy.update("g", "cheap", 0, "easy")
        expected = 0.95 + (0 - 30 * 0.95) / (30 + 10)
        self.assertAlmostEqual(policy.centre("g", "cheap", "easy"), expected)
        self.assertAlmostEqual(policy.centre("other", "cheap", "easy"), 0.95)

    def test_draws_centre_on_prediction_without_group_data(self):
        policy = PriorPrimalDualThompson(ARMS, COSTS, 0.8, PRIORS, kappa=20, seed=3)
        draws = [policy.sample("g", "strong", "hard") for _ in range(3000)]
        self.assertAlmostEqual(statistics.mean(draws), 0.9, delta=0.01)
        self.assertAlmostEqual(statistics.stdev(draws), math.sqrt(0.9 * 0.1 / 21), delta=0.01)

    def test_shortfall_updates_lambda(self):
        policy = PriorPrimalDualThompson(ARMS, COSTS, 0.8, PRIORS, eta=1)
        policy.update("g", "cheap", 0, "hard")
        self.assertAlmostEqual(policy.accuracy_price, 0.8)


if __name__ == "__main__":
    unittest.main()


class RunnerOptionTests(unittest.TestCase):
    def test_priors_and_groups_files_are_read(self):
        import csv
        import tempfile
        from pathlib import Path
        from bandit.domain_compare import prior_factories, read_groups, read_priors
        with tempfile.TemporaryDirectory() as folder:
            priors = Path(folder) / "priors.csv"
            with priors.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["question_id", "predictor", "cheap", "strong"])
                writer.writerow(["q1", "text", 0.4, 0.9])
                writer.writerow(["q1", "subject", 0.7, 0.8])
            groups = Path(folder) / "groups.csv"
            groups.write_text("question_id,group\nq1,g03\n")
            table = read_priors(priors, ["text"], ARMS)
            self.assertEqual(table, {"text": {"q1": {"cheap": 0.4, "strong": 0.9}}})
            self.assertEqual(read_groups(groups), {"q1": "g03"})
            with self.assertRaises(ValueError):
                read_priors(priors, ["missing"], ARMS)
            factories = prior_factories(table, [20, math.inf], 1.0, 0.0)
            self.assertEqual(set(factories), {"prior_text_k20", "prior_text_kinf"})
            policy = factories["prior_text_kinf"](ARMS, COSTS, 0.8, "ts:1")
            self.assertEqual(policy.sample("g", "strong", "q1"), 0.9)
