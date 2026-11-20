"""Algorithm 6 (FT-UCB), Appendix H of Juneja et al., ICLR 2025."""

import math
import random


class FixedThresholdUCB:
    """Use a fixed known cost vector and update only the sampled arm's reward.

    Arms are indexed by ascending cost (then name). The first K decisions sample
    each arm once in that order. Equal-cost ties subsequently use this same order.
    UCB values are deliberately not clipped to 1 or replaced with KL-UCB bounds.
    """

    def __init__(self, arms, costs, threshold, seed=0):
        arms = tuple(arms)
        if not arms or len(set(arms)) != len(arms):
            raise ValueError("FT-UCB needs nonempty, unique arms")
        if set(costs) != set(arms):
            raise ValueError("FT-UCB costs must contain exactly one entry per arm")
        if any(not math.isfinite(c) or c < 0 for c in costs.values()):
            raise ValueError("FT-UCB costs must be finite and nonnegative")
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("FT-UCB needs a threshold in (0, 1]")
        self.arms = tuple(sorted(arms, key=lambda arm: (costs[arm], arm)))
        self.threshold = threshold
        self.counts = dict.fromkeys(self.arms, 0)
        self.successes = dict.fromkeys(self.arms, 0)
        self.rng = random.Random(seed)
        self.last_decision = None

    def select(self):
        t = sum(self.counts.values()) + 1
        if t <= len(self.arms):
            arm = self.arms[t - 1]
            self.last_decision = {"phase": "initialization", "ucb": None, "candidates": []}
            return arm
        bounds = {
            arm: self.successes[arm] / self.counts[arm]
            + math.sqrt(2 * math.log(t) / self.counts[arm])
            for arm in self.arms
        }
        candidates = [arm for arm in self.arms if bounds[arm] >= self.threshold]
        arm = candidates[0] if candidates else self.rng.choice(self.arms)
        self.last_decision = {
            "phase": "feasible_min_cost" if candidates else "uniform_fallback",
            "ucb": bounds[arm], "candidates": candidates,
        }
        return arm

    def update(self, arm, correct):
        if arm not in self.counts:
            raise ValueError(f"Unknown arm: {arm}")
        if correct not in (0, 1):
            raise ValueError("FT-UCB needs correct to be 0 or 1")
        self.counts[arm] += 1
        self.successes[arm] += correct
