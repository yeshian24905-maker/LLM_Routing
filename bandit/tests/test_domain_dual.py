"""Hand-computed checks for the domain-aware primal-dual policy, its oracle and its runner."""

import math
import unittest

from bandit.data import Observation
from bandit.domain_compare import policy_factories, run_domain_comparison
from bandit.domain_dual import DomainPrimalDualUCB
from bandit.domain_oracle import oracle_routing


ARMS = ("cheap", "strong")
COSTS = {"cheap": 1.0, "strong": 4.0}


class EstimateTests(unittest.TestCase):
    def test_unseen_domain_uses_pooled_estimate(self):
        policy = DomainPrimalDualUCB(ARMS, COSTS, 0.8, kappa=20)
        for correct in (1, 1, 1, 0):
            policy.update("a", "cheap", correct)
        pooled = 4 / 6
        pooled_var = pooled * (1 - pooled) / 7
        mean, sd = policy.estimate("b", "cheap")
        self.assertAlmostEqual(mean, pooled)
        self.assertAlmostEqual(sd, math.sqrt(pooled * (1 - pooled) / 21 + pooled_var))

    def test_domain_data_shrinks_toward_pooled_estimate(self):
        policy = DomainPrimalDualUCB(ARMS, COSTS, 0.8, kappa=4)
        for correct in (1, 1, 1, 1):
            policy.update("a", "cheap", correct)
        for correct in (0, 0):
            policy.update("b", "cheap", correct)
        pooled = 5 / 8
        mean, _ = policy.estimate("a", "cheap")
        self.assertAlmostEqual(mean, (4 + 4 * pooled) / 8)

    def test_infinite_kappa_ignores_domains(self):
        policy = DomainPrimalDualUCB(ARMS, COSTS, 0.8, kappa=math.inf)
        policy.update("a", "cheap", 1)
        policy.update("b", "cheap", 0)
        self.assertEqual(policy.estimate("a", "cheap"), policy.estimate("b", "cheap"))
        self.assertAlmostEqual(policy.estimate("a", "cheap")[0], 0.5)


class SelectionTests(unittest.TestCase):
    def test_zero_price_picks_cheapest_and_ties_go_cheaper(self):
        policy = DomainPrimalDualUCB(ARMS, COSTS, 0.8)
        self.assertEqual(policy.select("a"), "cheap")
        free = DomainPrimalDualUCB(ARMS, {"cheap": 0.0, "strong": 0.0}, 0.8)
        self.assertEqual(free.select("a"), "cheap")

    def test_shortfall_raises_price_until_stronger_arm_wins(self):
        policy = DomainPrimalDualUCB(ARMS, COSTS, 0.8, kappa=1, beta=0, eta=1)
        for _ in range(10):
            policy.update("a", "strong", 1)
        for _ in range(10):
            policy.update("a", "cheap", 0)
        self.assertAlmostEqual(policy.shortfall, 20 * 0.8 - 10)
        self.assertAlmostEqual(policy.accuracy_price, 6.0)
        self.assertEqual(policy.select("a"), "strong")

    def test_surplus_clips_price_at_zero(self):
        policy = DomainPrimalDualUCB(ARMS, COSTS, 0.5, eta=1)
        for _ in range(5):
            policy.update("a", "cheap", 1)
        self.assertEqual(policy.accuracy_price, 0.0)

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            DomainPrimalDualUCB(ARMS, {"cheap": 1.0}, 0.8)
        with self.assertRaises(ValueError):
            DomainPrimalDualUCB(ARMS, COSTS, 0.8, kappa=0)
        policy = DomainPrimalDualUCB(ARMS, COSTS, 0.8)
        with self.assertRaises(ValueError):
            policy.update("a", "other", 1)
        with self.assertRaises(ValueError):
            policy.update("a", "cheap", 2)


def fixture():
    """Domain x: cheap 50%, strong 100%. Domain y: both 100%. Four questions each."""
    table = {}
    for i in range(8):
        domain = "x" if i < 4 else "y"
        cheap = int(domain == "y" or i % 2 == 0)
        table[f"q{i}"] = {"cheap": Observation(f"q{i}", "cheap", cheap, 1.0),
                          "strong": Observation(f"q{i}", "strong", 1, 4.0)}
    return table, {f"q{i}": "x" if i < 4 else "y" for i in range(8)}


class OracleTests(unittest.TestCase):
    def test_domain_oracle_upgrades_only_the_domain_that_gains(self):
        table, domains = fixture()
        # Cheap everywhere gives 0.75. Reaching 0.875 needs half of domain x on strong.
        result = oracle_routing(table, domains, ARMS, COSTS, 0.875)
        self.assertAlmostEqual(result["cost_usd"], 0.5 * 1.0 + 0.5 * (0.5 * 1.0 + 0.5 * 4.0))
        self.assertAlmostEqual(result["accuracy"], 0.875)

    def test_global_oracle_mixes_one_arm_for_all_questions(self):
        table, domains = fixture()
        result = oracle_routing(table, domains, ARMS, COSTS, 0.875, per_domain=False)
        self.assertAlmostEqual(result["cost_usd"], 2.5)

    def test_unreachable_threshold_reports_no_cost(self):
        table, domains = fixture()
        table["q0"]["strong"] = Observation("q0", "strong", 0, 4.0)
        result = oracle_routing(table, domains, ARMS, COSTS, 1.0)
        self.assertFalse(result["feasible"])
        self.assertIsNone(result["cost_usd"])


class RunnerTests(unittest.TestCase):
    def test_runner_is_paired_and_reports_every_policy(self):
        table, domains = fixture()
        factories = policy_factories([math.inf, 20], [1.0], [0.5], 0.0)
        runs, aggregates, curves = run_domain_comparison(
            table, domains, ARMS, COSTS, factories, horizons=[8], thresholds=[0.875],
            repeats=2, seed=1, checkpoint=4)
        self.assertEqual(len(runs), 2 * len(factories))
        self.assertEqual({a["policy"] for a in aggregates}, set(factories))
        self.assertEqual(len(curves), 2 * len(factories))
        for run in runs:
            self.assertAlmostEqual(sum(run["model_share"].values()), 1.0)
        self.assertAlmostEqual(aggregates[0]["oracle_domain_cost_usd_per_1k"], 1750.0)


if __name__ == "__main__":
    unittest.main()
