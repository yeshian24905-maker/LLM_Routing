"""Hand-computed checks for Algorithm 6 and the two regret definitions.

Run explicitly with: python -m unittest discover -s bandit/tests
"""

import csv
import hashlib
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from bandit.data import REQUIRED_FIELDS, Observation, grade, load_data, load_questions
from bandit.ft_evaluate import arm_statistics, evaluate_ft, resolve_costs
from bandit.ft_report import make_ft_report
from bandit.ft_ucb import FixedThresholdUCB


ARMS = ["cheap", "mid", "strong", "premium"]
COSTS = dict(zip(ARMS, (1.0, 2.0, 3.0, 4.0)))


def fixture(all_wrong=False, missing_cost=False):
    return {
        f"q{i}": {
            arm: Observation(
                f"q{i}", arm,
                0 if all_wrong else int(arm in ("strong", "premium") or (arm == "mid" and i < 4)),
                None if missing_cost and i == 0 else 10.0,
            )
            for arm in ARMS
        }
        for i in range(8)
    }


class FixedThresholdPolicyTests(unittest.TestCase):
    def test_initial_samples_follow_cost_order(self):
        policy = FixedThresholdUCB(list(reversed(ARMS)), COSTS, 0.85)
        for expected in ARMS:
            self.assertEqual(policy.select(), expected)
            self.assertEqual(policy.last_decision["phase"], "initialization")
            policy.update(expected, 1)
        self.assertEqual(policy.select(), "cheap")
        self.assertGreater(policy.last_decision["ucb"], 1.0)

    def test_choose_cheapest_qualifying_arm_instead_of_highest_ucb(self):
        policy = FixedThresholdUCB(ARMS, COSTS, 0.85)
        policy.counts = dict.fromkeys(ARMS, 1000)
        policy.successes = dict(zip(ARMS, (600, 850, 930, 960)))
        self.assertEqual(policy.select(), "mid")
        self.assertEqual(policy.last_decision["candidates"], ["mid", "strong", "premium"])
        self.assertAlmostEqual(policy.last_decision["ucb"], 0.85 + math.sqrt(2 * math.log(4001) / 1000))

    def test_empty_candidate_set_uses_seeded_uniform_choice(self):
        policy = FixedThresholdUCB(ARMS, COSTS, 0.85, seed=123)
        policy.counts = dict.fromkeys(ARMS, 1000)
        expected = random.Random(123).choice(policy.arms)
        self.assertEqual(policy.select(), expected)
        self.assertEqual(policy.last_decision["phase"], "uniform_fallback")
        self.assertEqual(policy.last_decision["candidates"], [])

    def test_only_updated_arm_changes(self):
        policy = FixedThresholdUCB(ARMS, COSTS, 0.85)
        policy.update("cheap", 1)
        self.assertEqual(policy.counts, dict(zip(ARMS, (1, 0, 0, 0))))
        self.assertEqual(policy.successes, dict(zip(ARMS, (1, 0, 0, 0))))

    def test_invalid_costs_and_thresholds(self):
        for costs in ({"cheap": 1}, {**COSTS, "cheap": -1}, {**COSTS, "cheap": math.nan}):
            with self.subTest(costs=costs), self.assertRaises(ValueError):
                FixedThresholdUCB(ARMS, costs, 0.85)
        for threshold in (0, 1.01, math.nan):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                FixedThresholdUCB(ARMS, COSTS, threshold)


class FixedThresholdReplayTests(unittest.TestCase):
    def test_hand_computed_regrets_and_realized_cost(self):
        summary, runs, traces, curves = evaluate_ft(
            fixture(), ARMS, thresholds=[0.8], repeats=1, fixed_costs=COSTS,
        )
        run = runs[0]
        # Four initial pulls, then four extra cheap-arm pulls.
        self.assertEqual(run["model_selections"], dict(zip(ARMS, (5, 1, 1, 1))))
        self.assertEqual(run["optimal_feasible_model"], "strong")
        self.assertEqual(run["cost_regret_usd"], 1.0)  # only premium costs more than a*
        self.assertAlmostEqual(run["quality_regret"], 5 * 0.8 + 1 * 0.3)
        self.assertEqual(run["total_cost_usd"], 80.0)
        self.assertEqual(run["fixed_cost_total_usd"], 14.0)
        self.assertEqual(curves[-1]["cost_regret_mean_usd"], run["cost_regret_usd"])
        self.assertEqual(curves[-1]["quality_regret_mean"], run["quality_regret"])
        self.assertEqual(summary["arm_statistics"]["mid"]["accuracy"], 0.5)
        self.assertEqual(sum(run["model_selections"].values()), 8)
        self.assertEqual(len({row["question_id"] for row in traces}), 8)
        for field in ("cost_regret_usd", "quality_regret"):
            values = [row[field] for row in traces]
            self.assertEqual(values, sorted(values))

    def test_infeasible_threshold_has_undefined_cost_regret(self):
        summary, runs, traces, curves = evaluate_ft(
            fixture(all_wrong=True), ARMS, thresholds=[0.8], repeats=1, fixed_costs=COSTS,
        )
        self.assertTrue(summary["scenarios"][0]["infeasible"])
        self.assertIsNone(runs[0]["cost_regret_usd"])
        self.assertTrue(all(row["cost_regret_usd"] is None for row in traces))
        self.assertTrue(all(row["cost_regret_mean_usd"] is None for row in curves))
        self.assertAlmostEqual(runs[0]["quality_regret"], 8 * 0.8)

    def test_missing_actual_cost_does_not_become_zero(self):
        _, runs, _, _ = evaluate_ft(
            fixture(missing_cost=True), ARMS, thresholds=[0.8], repeats=1, fixed_costs=COSTS,
        )
        self.assertIsNone(runs[0]["total_cost_usd"])
        self.assertEqual(runs[0]["known_cost_usd"], 70.0)
        self.assertEqual(runs[0]["unknown_cost_count"], 1)
        self.assertEqual(runs[0]["fixed_cost_total_usd"], 14.0)

    def test_prompt_order_shared_and_runs_start_fresh(self):
        _, runs, traces, _ = evaluate_ft(
            fixture(), ARMS, thresholds=[0.8, 0.85], repeats=2, fixed_costs=COSTS,
        )
        for seed in (42, 43):
            paths = [[row for row in traces if row["threshold"] == mu and row["seed"] == seed]
                     for mu in (0.8, 0.85)]
            self.assertEqual([row["question_id"] for row in paths[0]],
                             [row["question_id"] for row in paths[1]])
            for path in paths:
                self.assertEqual([row["step"] for row in path], list(range(1, 9)))
                self.assertEqual([row["model"] for row in path[:4]], ARMS)
                self.assertEqual([row["prior_arm_count"] for row in path[:4]], [0] * 4)
        self.assertTrue(all(sum(run["model_selections"].values()) == 8 for run in runs))

    def test_evaluation_labels_of_unselected_arms_do_not_change_routing(self):
        table = fixture()
        # These arms are never sampled after initialization in this short horizon.
        order = sorted(table)
        random.Random("order:42").shuffle(order)
        modified = {qid: dict(observations) for qid, observations in table.items()}
        for question_id in order[4:]:
            for arm in ("mid", "strong", "premium"):
                row = modified[question_id][arm]
                modified[question_id][arm] = Observation(row.question_id, arm, 1 - row.correct, row.cost)
        outputs = [evaluate_ft(data, ARMS, [0.8], repeats=1, fixed_costs=COSTS)[2]
                   for data in (table, modified)]
        self.assertEqual([row["model"] for row in outputs[0]], [row["model"] for row in outputs[1]])

    def test_cost_source_must_be_explicit(self):
        stats = arm_statistics(fixture(missing_cost=True), ARMS)
        with self.assertRaises(ValueError):
            resolve_costs(ARMS, stats)
        with self.assertRaises(ValueError):
            resolve_costs(ARMS, stats, COSTS, empirical_costs=True)
        self.assertEqual(resolve_costs(ARMS, stats, empirical_costs=True), dict.fromkeys(ARMS, 10.0))

    def test_experiment_requires_four_arms(self):
        with self.assertRaises(ValueError):
            evaluate_ft(fixture(), ARMS[:3], [0.8], repeats=1, fixed_costs=COSTS)

    def test_report_includes_comparator_counts_and_cost_assumptions(self):
        summary, runs, _, _ = evaluate_ft(
            fixture(), ARMS, [0.8], repeats=1, fixed_costs=COSTS,
        )
        summary.update(input_csv="fixture.csv", input_sha256="test", arm_costs_json="costs.json",
                       arm_costs_sha256="test", questions_jsonl="questions.jsonl", questions_sha256="test",
                       scorer="test", grading={"needs_review": 0})
        report = make_ft_report(summary, runs)
        for expected in ("5 | 1 | 1 | 1 | 8", "Quality regret", "given fixed costs", "regret.png"):
            self.assertIn(expected, report)


class MMMUInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.questions = [
            {"question_id": qid, "subject": "Math", "question_type": "multiple-choice",
             "question": "Choose A.", "correct_answer": "A", "images": {},
             "options": ["First", "Second", "Third", "Fourth"]}
            for qid in ("q2", "q1")
        ]
        raw = "".join(json.dumps(question) + "\n" for question in self.questions).encode()
        (self.root / "questions.jsonl").write_bytes(raw)
        (self.root / "dataset_manifest.json").write_text(json.dumps({
            "questions_sha256": hashlib.sha256(raw).hexdigest(), "question_count": 2,
        }), encoding="utf-8")
        self.rows = [
            {"question_id": question["question_id"], "subject": "Math", "question_type": "multiple-choice",
             "model": arm, "raw_response": "<answer>A</answer>", "correct_answer": "A", "cost": "0.01", "error": ""}
            for question in reversed(self.questions) for arm in ARMS
        ]
        self.path = self.root / "results.csv"
        self.write_results()

    def write_results(self):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REQUIRED_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

    def test_questions_determine_round_count_and_base_order(self):
        questions = load_questions(self.root, expected_prompts=2)
        table, models, scored = load_data(self.path, questions)
        self.assertEqual(list(table), ["q2", "q1"])
        self.assertEqual(len(table), 2)  # two prompts, not eight decision rounds
        self.assertEqual(len(scored), 8)
        self.assertEqual(set(models), set(ARMS))

    def test_result_cannot_override_mmmu_ground_truth(self):
        self.rows[0]["correct_answer"] = "B"
        self.write_results()
        with self.assertRaisesRegex(ValueError, "does not match MMMU"):
            load_data(self.path, load_questions(self.root, 2))

    def test_unknown_question_is_rejected(self):
        self.rows[0]["question_id"] = "not-in-mmmu"
        self.write_results()
        with self.assertRaisesRegex(ValueError, "outside questions.jsonl"):
            load_data(self.path, load_questions(self.root, 2))

    def test_manifest_detects_changed_question_file(self):
        with (self.root / "questions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_questions(self.root, 2)

    def test_missing_arm_outcome_is_rejected(self):
        self.rows.pop()
        self.write_results()
        with self.assertRaisesRegex(ValueError, "Missing model results"):
            load_data(self.path, load_questions(self.root, 2))


class ChoiceScoringTests(unittest.TestCase):
    def score(self, response, gold="B", options=("49", "147", "196")):
        return grade({"raw_response": response, "question_type": "multiple-choice",
                      "correct_answer": gold, "error": ""}, options)

    def test_labeled_option_with_explanation_is_recognized(self):
        self.assertEqual(self.score("<answer>B. 147 units</answer>")[2], 1)

    def test_option_value_without_letter_is_recognized(self):
        self.assertEqual(self.score("<answer>147</answer>")[0], "B")
        self.assertEqual(self.score("<answer>147</answer>")[2], 1)

    def test_option_mapping_is_independent_of_gold(self):
        prediction, _, correct = self.score("<answer>147</answer>", gold="A")
        self.assertEqual(prediction, "B")
        self.assertEqual(correct, 0)

    def test_multi_option_answer_remains_ambiguous(self):
        self.assertEqual(self.score("<answer>B, C</answer>")[0], "")
        self.assertEqual(self.score("<answer>B, C</answer>")[2], 0)

    def test_duplicate_option_text_is_not_guessed(self):
        self.assertEqual(self.score("<answer>147</answer>", options=("147", "147"))[0], "")

    def test_last_answer_tag_takes_precedence(self):
        self.assertEqual(self.score("<answer>B</answer> revised: <answer>A</answer>")[2], 0)

    def test_reasoning_outside_answer_tag_cannot_rescue_bad_answer(self):
        self.assertEqual(self.score("<answer>no match</answer> Reasoning mentions option B.")[2], 0)


if __name__ == "__main__":
    unittest.main()
