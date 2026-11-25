"""Primal-dual Thompson sampling around per-question accuracy predictions."""

import math

from .domain_thompson import DomainPrimalDualThompson


class PriorPrimalDualThompson(DomainPrimalDualThompson):
    """Centre each draw on an offline prediction p(question, arm) plus a learned group offset.

    The offset of an arm in a group is (successes - predicted successes) / (n + kappa),
    so a group follows the predictions until its outcomes show a bias. The draw is
    Beta(c * (kappa + n), (1 - c) * (kappa + n)) around the centre c; with kappa = inf
    the policy uses the predictions as they are. lambda follows DomainPrimalDualUCB.
    """

    def __init__(self, arms, costs, threshold, priors, kappa=20.0, eta=1.0, lambda0=0.0, seed=0):
        """priors: {question_id: {arm: predicted accuracy}}, each from a fit that never saw that question."""
        super().__init__(arms, costs, threshold, kappa=kappa, eta=eta, lambda0=lambda0, seed=seed)
        self.priors = priors
        self.predicted_successes = {}

    def centre(self, domain, arm, question):
        prediction = self.priors[question][arm]
        if math.isinf(self.kappa):
            return prediction
        key = domain, arm
        n = self.counts.get(key, 0)
        offset = (self.successes.get(key, 0) - self.predicted_successes.get(key, 0.0)) / (n + self.kappa)
        return min(max(prediction + offset, 1e-3), 1 - 1e-3)

    def sample(self, domain, arm, question):
        centre = self.centre(domain, arm, question)
        if math.isinf(self.kappa):
            return centre
        strength = self.kappa + self.counts.get((domain, arm), 0)
        return self.rng.betavariate(centre * strength, (1 - centre) * strength)

    def select(self, domain, question=None):
        self.t += 1
        price = self.accuracy_price
        draws = {arm: self.sample(domain, arm, question) for arm in self.arms}
        arm = min(self.arms, key=lambda a: self.prices[a] - price * draws[a])
        self.last_decision = {"accuracy_price": price, "draws": draws}
        return arm

    def update(self, domain, arm, correct, question=None):
        super().update(domain, arm, correct)
        key = domain, arm
        self.predicted_successes[key] = self.predicted_successes.get(key, 0.0) + self.priors[question][arm]
