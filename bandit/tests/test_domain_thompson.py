"""Checks for the domain-aware primal-dual Thompson sampling policy."""

import math
import statistics
import unittest

from bandit.domain_thompson import DomainPrimalDualThompson


ARMS = ("cheap", "strong")
COSTS = {"cheap": 1.0, "strong": 4.0}


class ThompsonTests(unittest.TestCase):
    def test_same_seed_gives_same_choices(self):
        def choices(seed):
            policy = DomainPrimalDualThompson(ARMS, COSTS, 0.8, lambda0=5.0, seed=seed)
            return [policy.select("a") for _ in range(50)]
        self.assertEqual(choices(3), choices(3))

    def test_zero_price_picks_cheapest(self):
        policy = DomainPrimalDualThompson(ARMS, COSTS, 0.8, seed=1)
        self.assertTrue(all(policy.select("a") == "cheap" for _ in range(20)))

    def test_draws_follow_domain_data_when_it_dominates_the_prior(self):
        policy = DomainPrimalDualThompson(ARMS, COSTS, 0.8, kappa=2, seed=1)
        for _ in range(200):
            policy.update("a", "cheap", 0)
            policy.update("b", "cheap", 1)
        draws_a = [policy.sample("a", "cheap") for _ in range(200)]
        draws_b = [policy.sample("b", "cheap") for _ in range(200)]
        self.assertLess(statistics.mean(draws_a), 0.05)
        self.assertGreater(statistics.mean(draws_b), 0.95)

    def test_unseen_domain_draws_center_on_pooled_accuracy(self):
        policy = DomainPrimalDualThompson(ARMS, COSTS, 0.8, kappa=20, seed=2)
        for i in range(400):
            policy.update("a", "cheap", int(i % 4 != 0))
        draws = [policy.sample("new", "cheap") for _ in range(2000)]
        self.assertAlmostEqual(statistics.mean(draws), 0.75, delta=0.02)
        # Prior sd sqrt(0.75 * 0.25 / 21) ~ 0.094: a new domain may differ from the pooled value.
        self.assertAlmostEqual(statistics.stdev(draws), math.sqrt(0.75 * 0.25 / 21), delta=0.015)

    def test_infinite_kappa_ignores_domains(self):
        policy = DomainPrimalDualThompson(ARMS, COSTS, 0.8, kappa=math.inf, seed=4)
        for _ in range(300):
            policy.update("a", "cheap", 0)
        self.assertLess(statistics.mean(policy.sample("b", "cheap") for _ in range(200)), 0.02)

    def test_high_price_picks_the_arm_that_is_clearly_better(self):
        policy = DomainPrimalDualThompson(ARMS, COSTS, 0.8, kappa=1, lambda0=50.0, seed=5)
        for _ in range(100):
            policy.update("a", "cheap", 0)
            policy.update("a", "strong", 1)
        self.assertTrue(all(policy.select("a") == "strong" for _ in range(20)))


if __name__ == "__main__":
    unittest.main()
