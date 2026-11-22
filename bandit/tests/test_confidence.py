"""Checks for confidence formulas and fair, strictly offline paired comparisons."""

import math
import random
import socket
import unittest

from bandit.compare import network_disabled, prompt_stream, run_comparison
from bandit.confidence import POLICIES, FixedThresholdConfidence, bernoulli_kl, kl_upper_bound
from bandit.data import Observation
from bandit.ft_ucb import FixedThresholdUCB


ARMS = ("cheap", "mid", "strong", "premium")
COSTS = dict(zip(ARMS, (1.0, 2.0, 3.0, 4.0)))


class TraceCollector:
    def __init__(self):
        self.rows = []

    def writerow(self, row):
        self.rows.append(row)


def fixture():
    return {f"q{i}": {arm: Observation(f"q{i}", arm,
        int(arm in ("strong", "premium") or (arm == "mid" and i < 4)), 10.0)
        for arm in ARMS} for i in range(8)}


class BernoulliKLTests(unittest.TestCase):
    def test_endpoints_and_identical_parameters(self):
        self.assertEqual(bernoulli_kl(0, 0), 0)
        self.assertEqual(bernoulli_kl(1, 1), 0)
        self.assertTrue(math.isinf(bernoulli_kl(.5, 0)))
        self.assertTrue(math.isinf(bernoulli_kl(.5, 1)))
        self.assertAlmostEqual(bernoulli_kl(0, .8), -math.log(.2))
        self.assertAlmostEqual(bernoulli_kl(1, .8), -math.log(.8))

    def test_known_bernoulli_divergence(self):
        self.assertAlmostEqual(bernoulli_kl(.5, .8), .5*math.log(.5/.8)+.5*math.log(.5/.2))

    def test_root_satisfies_constraint_and_is_maximal(self):
        upper = kl_upper_bound(.8, 100, math.log(1000))
        self.assertGreater(upper, .8)
        self.assertAlmostEqual(100*bernoulli_kl(.8, upper), math.log(1000), places=11)
        self.assertGreater(100*bernoulli_kl(.8, upper+1e-7), math.log(1000))

    def test_upper_bound_endpoints(self):
        self.assertEqual(kl_upper_bound(1, 10, 2), 1)
        self.assertEqual(kl_upper_bound(.8, 10, 0), .8)
        self.assertAlmostEqual(kl_upper_bound(0, 10, 2), 1-math.exp(-.2))

    def test_kl_is_at_least_as_tight_as_quadratic_for_same_budget(self):
        for p in (0, .2, .5, .8, .99, 1):
            for n in (1, 10, 1000):
                upper = kl_upper_bound(p, n, 3)
                self.assertLessEqual(upper, min(1, p+math.sqrt(3/(2*n)))+1e-12)

    def test_invalid_inputs(self):
        for p, q in ((-.1, .5), (.5, 1.1), (math.nan, .5)):
            with self.assertRaises(ValueError):
                bernoulli_kl(p, q)
        for p, n, budget in ((.5, 0, 1), (.5, 1, -1), (.5, 1, math.inf)):
            with self.assertRaises(ValueError):
                kl_upper_bound(p, n, budget)


class ConfidencePolicyTests(unittest.TestCase):
    def test_original_configuration_exactly_matches_existing_policy(self):
        old = FixedThresholdUCB(ARMS, COSTS, .85, seed=123)
        new = FixedThresholdConfidence(ARMS, COSTS, .85, seed=123)
        rng = random.Random(12)
        for _ in range(500):
            chosen = old.select()
            self.assertEqual(chosen, new.select())
            self.assertEqual(old.last_decision['candidates'], new.last_decision['candidates'])
            self.assertEqual(old.last_decision['ucb'], new.last_decision['selected_bound'])
            reward = int(rng.random() < .8)
            old.update(chosen, reward)
            new.update(chosen, reward)

    def test_tighter_ucb_excludes_arm_before_original_does(self):
        policies = [FixedThresholdConfidence(ARMS, COSTS, .85, name)
                    for name in ("ucb_original", "ucb_tight")]
        for policy in policies:
            policy.counts = dict.fromkeys(ARMS, 1000)
            policy.successes = dict(zip(ARMS, (740, 800, 850, 900)))
        self.assertEqual(policies[0].select(), "cheap")
        self.assertEqual(policies[1].select(), "mid")
        self.assertAlmostEqual(policies[1].last_decision['selected_bound'],
                               .8+math.sqrt(.5*math.log(4001)/1000))

    def test_lcb_uses_minus_and_cheapest_eligible_not_highest_accuracy(self):
        policy = FixedThresholdConfidence(ARMS, COSTS, .8, 'lcb')
        policy.counts = dict.fromkeys(ARMS, 1000)
        policy.successes = dict(zip(ARMS, (700, 860, 950, 980)))
        self.assertEqual(policy.select(), 'strong')
        self.assertEqual(policy.last_decision['candidates'], ['strong', 'premium'])
        self.assertAlmostEqual(policy.last_decision['selected_bound'],
                               .95-math.sqrt(2*math.log(4001)/1000))

    def test_lcb_empty_set_keeps_uniform_fallback(self):
        policy = FixedThresholdConfidence(ARMS, COSTS, .89, 'lcb', seed=12)
        policy.counts = dict.fromkeys(ARMS, 1000)
        self.assertEqual(policy.select(), random.Random(12).choice(policy.arms))
        self.assertEqual(policy.last_decision['phase'], 'uniform_fallback')
        self.assertLess(policy.last_decision['selected_bound'], 0)

    def test_kl_threshold_test_matches_independent_root_solves(self):
        for name in ('kl_ucb', 'kl_ucb_c3'):
            policy = FixedThresholdConfidence(ARMS, COSTS, .85, name)
            policy.counts = dict.fromkeys(ARMS, 1000)
            policy.successes = dict(zip(ARMS, (740, 800, 850, 900)))
            t = 4001
            budget = math.log(t)+POLICIES[name]['loglog']*math.log(math.log(t))
            expected = [arm for arm in policy.arms if kl_upper_bound(
                policy.successes[arm]/1000, 1000, budget) >= .85]
            self.assertEqual(policy.select(), expected[0])
            self.assertEqual(policy.last_decision['candidates'], expected)

    def test_all_policies_initialize_once_and_update_only_chosen_arm(self):
        for name in POLICIES:
            policy = FixedThresholdConfidence(ARMS, COSTS, .85, name)
            for arm in ARMS:
                self.assertEqual(policy.select(), arm)
                self.assertEqual(policy.last_decision['phase'], 'initialization')
                before = dict(policy.counts)
                policy.update(arm, 1)
                self.assertEqual(policy.counts, {a: v+int(a == arm) for a, v in before.items()})


class PairedExperimentTests(unittest.TestCase):
    def test_3000_style_stream_uses_no_repeated_questions(self):
        stream, kind = prompt_stream(list(fixture()), 8, 42)
        expected = list(fixture())
        random.Random('order:42').shuffle(expected)
        self.assertEqual(stream, expected)
        self.assertEqual(len(set(stream)), 8)
        self.assertEqual(kind, 'without_replacement')

    def test_long_horizon_is_explicitly_resampled(self):
        stream, kind = prompt_stream(list(fixture()), 20, 42)
        self.assertEqual(len(stream), 20)
        self.assertLessEqual(len(set(stream)), 8)
        self.assertEqual(kind, 'with_replacement')
        self.assertEqual(stream, prompt_stream(list(fixture()), 20, 42)[0])

    def test_paired_trace_scope_and_hand_computed_regrets(self):
        collector = TraceCollector()
        runs, aggregates, curves = run_comparison(fixture(), ARMS, COSTS,
            policy_ids=['ucb_original', 'ucb_tight'], horizons=[8, 20], thresholds=[.8, .85],
            repeats=2, seed=42, trace_writer=collector)
        self.assertEqual(len(runs), 16)
        self.assertEqual(len(collector.rows), 2*2*(8+20))
        paths = [r for r in collector.rows if r['horizon'] == 8 and r['threshold'] == .8]
        for policy in ['ucb_original', 'ucb_tight']:
            rows = [r for r in paths if r['policy'] == policy]
            self.assertEqual([r['question_id'] for r in rows], prompt_stream(list(fixture()), 8, 42)[0])
            self.assertEqual(rows[-1]['cost_regret_usd'], 1.0)
            expected_quality = 5*.8+.3 if policy == 'ucb_original' else 2*.8+4*.3
            self.assertAlmostEqual(rows[-1]['quality_regret'], expected_quality)
            self.assertEqual(rows[-1]['known_cost_usd'], 80)
        baseline = [r for r in aggregates if r['policy'] == 'ucb_original']
        self.assertTrue(all(r['paired_accuracy_gain_pp'] == 0 for r in baseline))

    def test_unselected_outcomes_are_not_used_for_selection(self):
        source = fixture()
        first = TraceCollector()
        run_comparison(source, ARMS, COSTS, policy_ids=['ucb_original', 'lcb'],
            horizons=[8], thresholds=[.8], repeats=1, trace_writer=first)
        path = [r for r in first.rows if r['policy'] == 'lcb']
        changed = {qid: dict(rows) for qid, rows in source.items()}
        for row in path:
            for arm in ARMS:
                if arm != row['model']:
                    old = changed[row['question_id']][arm]
                    changed[row['question_id']][arm] = Observation(old.question_id, arm, 1-old.correct, old.cost)
        second = TraceCollector()
        run_comparison(changed, ARMS, COSTS, policy_ids=['ucb_original', 'lcb'],
            horizons=[8], thresholds=[.8], repeats=1, trace_writer=second)
        self.assertEqual([r['model'] for r in path], [r['model'] for r in second.rows if r['policy'] == 'lcb'])

    def test_network_guard_blocks_connections_and_restores_socket(self):
        original = socket.socket.connect
        with network_disabled() as attempts:
            with self.assertRaisesRegex(RuntimeError, 'disabled'):
                socket.create_connection(('example.invalid', 443))
            self.assertEqual(len(attempts), 1)
        self.assertIs(socket.socket.connect, original)


if __name__ == '__main__':
    unittest.main()
