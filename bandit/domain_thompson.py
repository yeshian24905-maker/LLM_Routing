"""Domain-aware primal-dual Thompson sampling under one global accuracy constraint."""

import math
import random

from .domain_dual import DomainPrimalDualUCB


class DomainPrimalDualThompson(DomainPrimalDualUCB):
    """Pick argmin_a price_a - lambda * theta(domain, a) with theta drawn from the posterior.

    The prior, the counts and the lambda update are those of DomainPrimalDualUCB;
    only the accuracy used for the choice differs. Each round draws the pooled
    accuracy m ~ Beta(1 + pooled successes, 1 + pooled failures), then
    theta ~ Beta(kappa * m + domain successes, kappa * (1 - m) + domain failures).
    """

    def __init__(self, arms, costs, threshold, kappa=20.0, eta=1.0, lambda0=0.0, seed=0):
        super().__init__(arms, costs, threshold, kappa=kappa, beta=0.0, eta=eta, lambda0=lambda0)
        self.rng = random.Random(seed)

    def sample(self, domain, arm):
        pooled_n, pooled_s = self.arm_counts[arm], self.arm_successes[arm]
        pooled = self.rng.betavariate(1 + pooled_s, 1 + pooled_n - pooled_s)
        if math.isinf(self.kappa):
            return pooled
        n = self.counts.get((domain, arm), 0)
        s = self.successes.get((domain, arm), 0)
        # A draw of exactly 0 or 1 would give a zero Beta parameter when the domain has no data.
        pooled = min(max(pooled, 1e-9), 1 - 1e-9)
        return self.rng.betavariate(self.kappa * pooled + s, self.kappa * (1 - pooled) + n - s)

    def select(self, domain, question=None):
        self.t += 1
        price = self.accuracy_price
        draws = {arm: self.sample(domain, arm) for arm in self.arms}
        arm = min(self.arms, key=lambda a: self.prices[a] - price * draws[a])
        self.last_decision = {"accuracy_price": price, "draws": draws}
        return arm
