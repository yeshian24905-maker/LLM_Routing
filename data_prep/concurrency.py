"""Automatic choice of how many requests to keep in flight under a requests-per-minute limit."""

import math
from collections import deque
from statistics import mean


class AutoConcurrency:
    """Keep enough requests in flight that the rate limit, not response time, sets the pace.

    By Little's law, requests in flight = request rate x time each request occupies a worker.
    The time is the mean over recently finished requests, so slower responses get more
    workers. The pacer still limits how often requests start, so extra workers cannot
    exceed the rate; they only wait for their turn.
    """

    def __init__(self, requests_per_minute, limit=32, window=20, initial_seconds=30.0):
        """
        requests_per_minute: rate the shared pacer allows.
        limit: upper bound on requests in flight, whatever the measured time.
        window: number of recent requests the mean time is taken over.
        initial_seconds: assumed time per request until the first one finishes.
        """
        self.rate = requests_per_minute / 60
        self.limit = limit
        self.initial_seconds = initial_seconds
        self.recent = deque(maxlen=window)

    def observe(self, seconds):
        self.recent.append(seconds)

    def mean_seconds(self):
        return mean(self.recent) if self.recent else self.initial_seconds

    def target(self):
        # One extra worker covers a request that is between its pacer slot and its POST.
        return max(1, min(self.limit, math.ceil(self.rate * self.mean_seconds()) + 1))
