"""KL-UCB routing that holds accuracy at a target and lowers the cost."""

import math
import random


def kl_bernoulli(p, q):
    eps = 1e-12
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def kl_upper_bound(mean, count, t):
    """Largest q with count * KL(mean, q) <= log(t) (Garivier and Cappe, 2011)."""
    bound = math.log(t) / count
    low, high = mean, 1.0
    for _ in range(40):
        middle = (low + high) / 2
        if kl_bernoulli(mean, middle) > bound:
            high = middle
        else:
            low = middle
    return low


class TargetUCB:
    """Pick the arm with the largest KL-UCB accuracy - mean cost / value.

    value is the USD value of one correct answer. After each answer,
    log(value) += step * (target - correct). The value rises while answers are
    below the target and falls while they are above it. Over T rounds this
    gives accuracy = target + log(initial_value / final_value) / (step * T),
    so a run meets the target when the final value is at most initial_value.
    The KL bound is used instead of the UCB1 bound because the UCB1 bound
    explores so much that it cannot reach targets near the best model.
    """

    def __init__(self, arms, target, initial_value=1.0, step=0.3, seed=0):
        self.arms = tuple(arms)
        if not self.arms or len(set(self.arms)) != len(self.arms):
            raise ValueError("TargetUCB needs nonempty, unique arms")
        if not 0 < target <= 1 or initial_value <= 0 or step <= 0:
            raise ValueError("TargetUCB needs 0 < target <= 1 and a positive initial value and step")
        self.target = target
        self.step = step
        self.log_value = math.log(initial_value)
        self.counts = dict.fromkeys(self.arms, 0)
        self.correct = dict.fromkeys(self.arms, 0)
        # Costs are averaged over known costs only, so they have their own count.
        self.cost_sums = dict.fromkeys(self.arms, 0.0)
        self.cost_counts = dict.fromkeys(self.arms, 0)
        self.rng = random.Random(seed)

    def mean_cost(self, arm):
        if self.cost_counts[arm]:
            return self.cost_sums[arm] / self.cost_counts[arm]
        # An arm without a known cost is assumed to be as expensive as the most expensive known arm.
        known = [self.cost_sums[a] / self.cost_counts[a] for a in self.arms if self.cost_counts[a]]
        return max(known, default=0.0)

    def select(self):
        untried = [arm for arm in self.arms if self.counts[arm] == 0]
        if untried:
            return self.rng.choice(untried)
        t = sum(self.counts.values()) + 1
        value = math.exp(self.log_value)
        scores = {
            arm: kl_upper_bound(self.correct[arm] / self.counts[arm], self.counts[arm], t)
            - self.mean_cost(arm) / value
            for arm in self.arms
        }
        best = max(scores.values())
        return self.rng.choice([arm for arm in self.arms if scores[arm] == best])

    def update(self, arm, correct, cost):
        if arm not in self.counts:
            raise ValueError(f"Unknown arm: {arm}")
        if correct not in (0, 1):
            raise ValueError("TargetUCB needs correct to be 0 or 1")
        self.counts[arm] += 1
        self.correct[arm] += correct
        if cost is not None:
            self.cost_sums[arm] += cost
            self.cost_counts[arm] += 1
        self.log_value += self.step * (self.target - correct)
