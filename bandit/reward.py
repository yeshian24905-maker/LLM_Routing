"""Rewards in [0, 1] that UCB1 maximizes."""


def accuracy_reward(observation):
    return observation.correct


def cost_reward(observation, value_per_correct):
    """Net value of one answer when a correct answer is worth value_per_correct USD.

    The net value is correct - cost / value_per_correct. The cost term is clipped
    to [0, 1] so that the net value stays in [-1, 1], and is then mapped to [0, 1]
    as UCB1 requires. An unknown cost gets the full penalty, so that a record
    without a cost never looks free.
    """
    if observation.cost is None:
        penalty = 1.0
    else:
        penalty = min(observation.cost / value_per_correct, 1.0)
    return (observation.correct - penalty + 1) / 2
