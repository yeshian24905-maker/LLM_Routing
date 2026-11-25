"""Domain-aware primal-dual UCB under one global accuracy constraint."""

import math


class DomainPrimalDualUCB:
    """Pick argmin_a price_a - lambda * UCB(domain, a) for each question.

    price_a is the arm cost divided by the highest arm cost. The accuracy of an
    arm in a domain has a Beta prior centred on the arm's pooled accuracy over
    all domains and worth `kappa` pseudo-observations, so a domain follows the
    pooled estimate until it has data of its own. lambda prices one unit of
    accuracy: it grows with the cumulative shortfall of realized correctness
    below the threshold and shrinks with the surplus. Ties go to the cheaper arm.
    """

    def __init__(self, arms, costs, threshold, kappa=20.0, beta=0.5, eta=1.0, lambda0=0.0):
        """kappa: prior weight in pseudo-observations; math.inf ignores domains.
        beta: the UCB radius is sqrt(beta * ln t) posterior standard deviations.
        eta: change of lambda per unit of cumulative accuracy shortfall.
        lambda0: lambda before any feedback.
        """
        arms = tuple(arms)
        if not arms or len(set(arms)) != len(arms) or set(costs) != set(arms):
            raise ValueError("Need unique arms with exactly one cost each")
        if any(not math.isfinite(c) or c < 0 for c in costs.values()):
            raise ValueError("Costs must be finite and nonnegative")
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("Threshold must lie in (0, 1]")
        if not kappa > 0 or beta < 0 or eta < 0 or lambda0 < 0 or not math.isfinite(beta + eta + lambda0):
            raise ValueError("Need kappa > 0 and finite, nonnegative beta, eta and lambda0")
        self.arms = tuple(sorted(arms, key=lambda arm: (costs[arm], arm)))
        top = max(costs.values())
        self.prices = {arm: costs[arm] / top if top > 0 else 0.0 for arm in self.arms}
        self.threshold, self.kappa, self.beta, self.eta, self.lambda0 = threshold, kappa, beta, eta, lambda0
        self.counts, self.successes = {}, {}
        self.arm_counts = dict.fromkeys(self.arms, 0)
        self.arm_successes = dict.fromkeys(self.arms, 0)
        self.shortfall = 0.0
        self.t = 0
        self.last_decision = None

    @property
    def accuracy_price(self):
        return max(0.0, self.lambda0 + self.eta * self.shortfall)

    def estimate(self, domain, arm):
        """Posterior mean and standard deviation of the arm's accuracy in the domain.

        The pooled accuracy has a Beta(1, 1) prior. Its own variance is added,
        scaled by the prior's share of the domain estimate, so an arm with
        little data anywhere stays uncertain in every domain.
        """
        pooled_n = self.arm_counts[arm]
        pooled = (self.arm_successes[arm] + 1) / (pooled_n + 2)
        pooled_var = pooled * (1 - pooled) / (pooled_n + 3)
        if math.isinf(self.kappa):
            return pooled, math.sqrt(pooled_var)
        n = self.counts.get((domain, arm), 0)
        mean = (self.successes.get((domain, arm), 0) + self.kappa * pooled) / (n + self.kappa)
        weight = self.kappa / (n + self.kappa)
        return mean, math.sqrt(mean * (1 - mean) / (n + self.kappa + 1) + weight**2 * pooled_var)

    def select(self, domain):
        self.t += 1
        radius = math.sqrt(self.beta * math.log(self.t))
        price = self.accuracy_price
        bounds = {}
        for arm in self.arms:
            mean, sd = self.estimate(domain, arm)
            bounds[arm] = mean + radius * sd
        arm = min(self.arms, key=lambda a: self.prices[a] - price * bounds[a])
        self.last_decision = {"accuracy_price": price, "bounds": bounds}
        return arm

    def update(self, domain, arm, correct):
        if arm not in self.arm_counts:
            raise ValueError(f"Unknown arm: {arm}")
        if correct not in (0, 1):
            raise ValueError("correct must be 0 or 1")
        key = domain, arm
        self.counts[key] = self.counts.get(key, 0) + 1
        self.successes[key] = self.successes.get(key, 0) + correct
        self.arm_counts[arm] += 1
        self.arm_successes[arm] += correct
        self.shortfall += self.threshold - correct
