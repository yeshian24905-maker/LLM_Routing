"""Replay each prompt once, revealing only the selected arm's result to the policy."""

import random
import statistics
from collections import Counter

from .reward import accuracy_reward, cost_reward
from .target_ucb import TargetUCB
from .ucb import UCB1


def cost_policy_name(value_per_correct):
    return f"ucb1-cost[V={value_per_correct:g}]"


def target_policy_name(target):
    return f"target-ucb[tau={target:g}]"


def select_ucb(table, models, order, seed, reward=accuracy_reward):
    policy = UCB1(models, seed=f"ucb:{seed}")
    selected = []
    for question_id in order:
        # Select before looking up this prompt's response or score.
        model = policy.select()
        observation = table[question_id][model]
        policy.update(model, reward(observation))
        selected.append(observation)
    return selected


def select_target(table, models, order, seed, target, initial_value, step):
    policy = TargetUCB(models, target, initial_value, step, seed=f"target:{seed}")
    selected = []
    for question_id in order:
        model = policy.select()
        observation = table[question_id][model]
        policy.update(model, observation.correct, observation.cost)
        selected.append(observation)
    return selected


def best_mix_cost(table, models, target):
    """Lowest expected USD cost of a fixed random mix of models with mean accuracy >= target.

    It uses every model's result on every prompt, so no router can rely on it.
    A mix of two models is enough because there is only one constraint.
    Returns None when no model reaches the target.
    """
    question_ids = list(table)
    accuracy, cost = {}, {}
    for model in models:
        rows = [table[qid][model] for qid in question_ids]
        accuracy[model] = sum(row.correct for row in rows) / len(rows)
        known = [row.cost for row in rows if row.cost is not None]
        cost[model] = statistics.mean(known) if known else float("inf")
    best = None
    for low in models:
        for high in models:
            if not accuracy[low] <= target <= accuracy[high]:
                continue
            spread = accuracy[high] - accuracy[low]
            share = (target - accuracy[low]) / spread if spread else 1.0
            mix = (1 - share) * cost[low] + share * cost[high]
            best = mix if best is None else min(best, mix)
    return None if best is None else best * len(question_ids)


def metrics(selected, best_fixed_reward, oracle_reward):
    correct = sum(row.correct for row in selected)
    known_latencies = [row.latency for row in selected if row.latency is not None]
    return {
        "prompts": len(selected),
        "correct": correct,
        "accuracy": correct / len(selected),
        "known_cost_usd": sum(row.cost for row in selected if row.cost is not None),
        "unknown_cost_count": sum(row.cost is None for row in selected),
        "mean_known_latency_seconds": statistics.mean(known_latencies) if known_latencies else None,
        "unknown_latency_count": len(selected) - len(known_latencies),
        "regret_vs_best_fixed": best_fixed_reward - correct,
        "gap_to_oracle": oracle_reward - correct,
        "model_selections": dict(sorted(Counter(row.model for row in selected).items())),
    }


def trace_rows(table, models, selected, seed, policy):
    cumulative_reward = 0
    known_cost = 0.0
    unknown_cost = 0
    fixed_rewards = dict.fromkeys(models, 0)
    oracle_reward = 0
    for step, row in enumerate(selected, 1):
        cumulative_reward += row.correct
        if row.cost is None:
            unknown_cost += 1
        else:
            known_cost += row.cost
        # These comparisons are computed after routing; they never update UCB1.
        observations = table[row.question_id]
        for model in models:
            fixed_rewards[model] += observations[model].correct
        oracle_reward += max(observation.correct for observation in observations.values())
        yield {
            "seed": seed, "policy": policy, "step": step,
            "question_id": row.question_id, "model": row.model, "correct": row.correct,
            "cost_usd": row.cost, "latency_seconds": row.latency,
            "cumulative_reward": cumulative_reward,
            "cumulative_accuracy": cumulative_reward / step,
            "known_cumulative_cost_usd": known_cost,
            "cumulative_unknown_cost_count": unknown_cost,
            "regret_vs_best_fixed": max(fixed_rewards.values()) - cumulative_reward,
            "gap_to_oracle": oracle_reward - cumulative_reward,
        }


def evaluate(
    table, models, repeats=30, seed=42, values_per_correct=(),
    targets=(), initial_value=1.0, step=0.3,
):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    question_ids = sorted(table)
    fixed_rewards = {
        model: sum(table[qid][model].correct for qid in question_ids) for model in models
    }
    best_fixed_reward = max(fixed_rewards.values())
    oracle_reward = sum(max(row.correct for row in table[qid].values()) for qid in question_ids)
    runs, traces = [], []
    for run_seed in range(seed, seed + repeats):
        order = question_ids.copy()
        random.Random(f"order:{run_seed}").shuffle(order)
        random_policy = random.Random(f"random:{run_seed}")
        selections = {"ucb1": select_ucb(table, models, order, run_seed)}
        for value in values_per_correct:
            selections[cost_policy_name(value)] = select_ucb(
                table, models, order, run_seed, lambda row, value=value: cost_reward(row, value),
            )
        for target in targets:
            selections[target_policy_name(target)] = select_target(
                table, models, order, run_seed, target, initial_value, step,
            )
        selections["random"] = [table[qid][random_policy.choice(models)] for qid in order]
        for model in models:
            selections[f"fixed:{model}"] = [table[qid][model] for qid in order]
        for name, selected in selections.items():
            runs.append({"seed": run_seed, "policy": name, **metrics(selected, best_fixed_reward, oracle_reward)})
            if not name.startswith("fixed:"):
                traces.extend(trace_rows(table, models, selected, run_seed, name))
    aggregate = {}
    for name in dict.fromkeys(row["policy"] for row in runs):
        rows = [row for row in runs if row["policy"] == name]
        aggregate[name] = {}
        for field in (
            "accuracy", "correct", "known_cost_usd", "unknown_cost_count",
            "mean_known_latency_seconds", "unknown_latency_count", "regret_vs_best_fixed", "gap_to_oracle",
        ):
            values = [row[field] for row in rows if row[field] is not None]
            aggregate[name][field] = {
                "mean": statistics.mean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
            }
        aggregate[name]["model_selections_mean"] = {
            model: statistics.mean(row["model_selections"].get(model, 0) for row in rows)
            for model in models
        }
    return {
        "prompts": len(table), "models": models, "repeats": repeats,
        "values_per_correct": list(values_per_correct),
        "targets": [
            {
                "policy": target_policy_name(target), "target": target,
                "runs_meeting_target": sum(
                    row["accuracy"] >= target for row in runs if row["policy"] == target_policy_name(target)
                ),
                "best_mix_cost_usd": best_mix_cost(table, models, target),
            }
            for target in targets
        ],
        "target_initial_value": initial_value, "target_step": step,
        "seeds": list(range(seed, seed + repeats)),
        "best_fixed_models": [model for model in models if fixed_rewards[model] == best_fixed_reward],
        "best_fixed_accuracy": best_fixed_reward / len(table),
        "oracle_accuracy": oracle_reward / len(table), "policies": aggregate,
    }, runs, traces
