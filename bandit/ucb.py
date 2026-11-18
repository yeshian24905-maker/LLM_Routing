"""Standard UCB1 with rewards bounded by [0, 1]."""

import math
import random


class UCB1:
    def __init__(self, arms, seed=0):
        self.arms = tuple(arms)
        if not self.arms or len(set(self.arms)) != len(self.arms):
            raise ValueError("UCB1 needs nonempty, unique arms")
        self.counts = dict.fromkeys(self.arms, 0)
        self.rewards = dict.fromkeys(self.arms, 0.0)
        self.rng = random.Random(seed)

    def select(self):
        untried = [arm for arm in self.arms if self.counts[arm] == 0]
        if untried:
            return self.rng.choice(untried)
        t = sum(self.counts.values()) + 1
        scores = {
            arm: self.rewards[arm] / self.counts[arm]
            + math.sqrt(2 * math.log(t) / self.counts[arm])
            for arm in self.arms
        }
        best = max(scores.values())
        return self.rng.choice([arm for arm in self.arms if scores[arm] == best])

    def update(self, arm, reward):
        if arm not in self.counts:
            raise ValueError(f"Unknown arm: {arm}")
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("UCB1 reward must be finite and in [0, 1]")
        self.counts[arm] += 1
        self.rewards[arm] += reward
