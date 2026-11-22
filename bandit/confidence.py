"""Fixed-threshold policies with Hoeffding or Bernoulli-KL confidence bounds.

Only the bound in Algorithm 6 is changed. Cost ordering, one initial pull per
arm, and uniform fallback for an empty candidate set remain the same.
"""

import math

from .ft_ucb import FixedThresholdUCB


POLICIES = {
    "ucb_original": {"label": "Original UCB", "kind": "ucb", "coefficient": 2.0},
    "ucb_tight": {"label": "Tight UCB", "kind": "ucb", "coefficient": 0.5},
    "kl_ucb": {"label": "KL-UCB (c=0)", "kind": "kl_ucb", "loglog": 0.0},
    "kl_ucb_c3": {"label": "KL-UCB (c=3)", "kind": "kl_ucb", "loglog": 3.0},
    "lcb": {"label": "LCB", "kind": "lcb", "coefficient": 2.0},
}


def bernoulli_kl(p, q):
    """KL(Bernoulli(p), Bernoulli(q)), including exact endpoints."""
    if not 0 <= p <= 1 or not 0 <= q <= 1:
        raise ValueError("Bernoulli parameters must lie in [0, 1]")
    if p == q:
        return 0.0
    if q in (0.0, 1.0):
        return math.inf
    if p == 0:
        return -math.log1p(-q)
    if p == 1:
        return -math.log(q)
    return max(0.0, p * math.log(p / q) + (1-p) * math.log((1-p) / (1-q)))


def kl_upper_bound(mean, n, budget):
    """Largest q >= mean satisfying n * KL(mean, q) <= budget."""
    if not 0 <= mean <= 1 or n <= 0 or not math.isfinite(budget) or budget < 0:
        raise ValueError("Invalid mean, sample count or KL budget")
    if mean == 1 or budget == 0:
        return mean
    if mean == 0:
        return -math.expm1(-budget / n)
    low, high = mean, 1.0
    for _ in range(52):
        mid = (low + high) / 2
        if bernoulli_kl(mean, mid) <= budget / n:
            low = mid
        else:
            high = mid
    return low


class FixedThresholdConfidence(FixedThresholdUCB):
    """Keep the original state/update contract, with configurable eligibility.

    For KL bounds, testing feasibility directly at mu0 is equivalent to solving
    the KL-UCB root for every arm. The selected root is computed only when a
    caller requests detailed diagnostics, avoiding repeated numerical solves.
    """

    def __init__(self, arms, costs, threshold, policy="ucb_original", seed=0):
        super().__init__(arms, costs, threshold, seed)
        if policy not in POLICIES:
            raise ValueError(f"Unknown confidence policy: {policy}")
        self.policy = policy
        self.spec = POLICIES[policy]

    def select(self, diagnostics=True):
        t = sum(self.counts.values()) + 1
        if t <= len(self.arms):
            self.last_decision = {
                "phase": "initialization", "selected_bound": None, "candidates": [],
            }
            return self.arms[t-1]
        means = {a: self.successes[a] / self.counts[a] for a in self.arms}
        log_t = math.log(t)
        if self.spec["kind"] == "kl_ucb":
            budget = log_t + self.spec["loglog"] * math.log(log_t)
            candidates = [a for a in self.arms if means[a] >= self.threshold or
                self.counts[a] * bernoulli_kl(means[a], self.threshold) <= budget]
            bounds = None
        else:
            sign = -1 if self.spec["kind"] == "lcb" else 1
            bounds = {a: means[a] + sign * math.sqrt(
                self.spec["coefficient"] * log_t / self.counts[a]) for a in self.arms}
            candidates = [a for a in self.arms if bounds[a] >= self.threshold]
        arm = candidates[0] if candidates else self.rng.choice(self.arms)
        selected_bound = None
        if diagnostics:
            selected_bound = (bounds[arm] if bounds is not None else
                kl_upper_bound(means[arm], self.counts[arm], budget))
        self.last_decision = {
            "phase": "feasible_min_cost" if candidates else "uniform_fallback",
            "selected_bound": selected_bound, "candidates": candidates,
        }
        return arm
