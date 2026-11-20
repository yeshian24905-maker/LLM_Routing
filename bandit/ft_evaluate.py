"""Offline FT-UCB replay and empirical versions of the paper's two regrets."""

import math
import random
import statistics

from .ft_ucb import FixedThresholdUCB


DEFAULT_THRESHOLDS = (0.89, 0.85, 0.80, 0.825)


def arm_statistics(table, models):
    result = {}
    for model in models:
        rows = [observations[model] for observations in table.values()]
        known = [row.cost for row in rows if row.cost is not None]
        result[model] = {
            "accuracy": sum(row.correct for row in rows) / len(rows),
            "mean_known_cost_usd": statistics.mean(known) if known else None,
            "unknown_cost_count": len(rows) - len(known),
        }
    return result


def resolve_costs(models, stats, fixed_costs=None, empirical_costs=False):
    """Require an explicit choice between given costs and offline calibration."""
    if (fixed_costs is not None) == empirical_costs:
        raise ValueError("Choose either --arm-costs JSON or --empirical-costs for FT-UCB")
    if fixed_costs is not None:
        if not isinstance(fixed_costs, dict) or set(fixed_costs) != set(models):
            raise ValueError("--arm-costs must be a JSON object with exactly the input model names")
        if any(isinstance(c, bool) or not isinstance(c, (int, float)) for c in fixed_costs.values()):
            raise ValueError("--arm-costs values must be numbers in USD per call")
        costs = dict(fixed_costs)
    else:
        costs = {model: stats[model]["mean_known_cost_usd"] for model in models}
        if any(cost is None for cost in costs.values()):
            raise ValueError("Cannot calibrate costs: at least one model has no known CSV costs")
    if any(not math.isfinite(cost) or cost < 0 for cost in costs.values()):
        raise ValueError("Arm costs must be finite and nonnegative")
    return costs


def moments(values):
    if not values or any(value is None for value in values):
        return {"mean": None, "std": None}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def evaluate_ft(
    table, models, thresholds=DEFAULT_THRESHOLDS, repeats=1, seed=42,
    fixed_costs=None, empirical_costs=False,
):
    if not table or repeats < 1:
        raise ValueError("FT-UCB needs a nonempty table and positive repeats")
    if len(models) != 4:
        raise ValueError("This FT-UCB experiment requires exactly four models")
    thresholds = tuple(thresholds)
    if not thresholds or len(set(thresholds)) != len(thresholds) or any(
        not math.isfinite(mu) or not 0 < mu <= 1 for mu in thresholds
    ):
        raise ValueError("Thresholds must be unique values in (0, 1]")
    stats = arm_statistics(table, models)
    costs = resolve_costs(models, stats, fixed_costs, empirical_costs)
    arm_order = sorted(models, key=lambda arm: (costs[arm], arm))
    runs, traces, curves, scenarios = [], [], [], []
    for threshold in thresholds:
        feasible = [arm for arm in arm_order if stats[arm]["accuracy"] >= threshold]
        optimal = feasible[0] if feasible else None
        scenario_runs, paths = [], []
        for run_seed in range(seed, seed + repeats):
            # load_data orders the table by the authoritative questions.jsonl list.
            order = list(table)
            # Same seed gives the same prompt order at every threshold.
            random.Random(f"order:{run_seed}").shuffle(order)
            policy = FixedThresholdUCB(models, costs, threshold, seed=f"ft:{run_seed}")
            correct, known_cost, unknown_costs, fixed_total = 0, 0.0, 0, 0.0
            cost_regret, quality_regret = (0.0 if optimal is not None else None), 0.0
            path = []
            for t, question_id in enumerate(order, 1):
                # Full-data accuracies below are for evaluation only, never selection.
                arm = policy.select()
                prior_count = policy.counts[arm]
                prior_mean = policy.successes[arm] / prior_count if prior_count else None
                observation = table[question_id][arm]
                policy.update(arm, observation.correct)
                correct += observation.correct
                unknown_costs += observation.cost is None
                if observation.cost is not None:
                    known_cost += observation.cost
                fixed_total += costs[arm]
                cost_gap = max(0.0, costs[arm] - costs[optimal]) if optimal is not None else None
                quality_gap = max(0.0, threshold - stats[arm]["accuracy"])
                if cost_gap is not None:
                    cost_regret += cost_gap
                quality_regret += quality_gap
                path.append({
                    "threshold": threshold, "seed": run_seed, "step": t,
                    "question_id": question_id, "model": arm,
                    "phase": policy.last_decision["phase"],
                    "prior_arm_count": prior_count, "prior_arm_mean": prior_mean,
                    "selected_ucb": policy.last_decision["ucb"],
                    "candidate_models": policy.last_decision["candidates"],
                    "correct": observation.correct, "cost_usd": observation.cost,
                    "fixed_arm_cost_usd": costs[arm],
                    "cumulative_correct": correct, "cumulative_accuracy": correct / t,
                    "known_cumulative_cost_usd": known_cost,
                    "cumulative_unknown_cost_count": unknown_costs,
                    "fixed_cumulative_cost_usd": fixed_total,
                    "cost_regret_increment_usd": cost_gap,
                    "quality_regret_increment": quality_gap,
                    "cost_regret_usd": cost_regret, "quality_regret": quality_regret,
                })
            run = {
                "threshold": threshold, "seed": run_seed, "prompts": len(order),
                "correct": correct, "accuracy": correct / len(order),
                "meets_threshold": correct / len(order) >= threshold,
                "total_cost_usd": known_cost if not unknown_costs else None,
                "known_cost_usd": known_cost, "unknown_cost_count": unknown_costs,
                "fixed_cost_total_usd": fixed_total, "optimal_feasible_model": optimal,
                "cost_regret_usd": cost_regret, "quality_regret": quality_regret,
                "model_selections": dict(policy.counts),
            }
            scenario_runs.append(run)
            runs.append(run)
            traces.extend(path)
            paths.append(path)
        for t, rows in enumerate(zip(*paths), 1):
            cost, quality = moments([row["cost_regret_usd"] for row in rows]), moments(
                [row["quality_regret"] for row in rows]
            )
            curves.append({
                "threshold": threshold, "step": t,
                "cost_regret_mean_usd": cost["mean"], "cost_regret_std_usd": cost["std"],
                "quality_regret_mean": quality["mean"], "quality_regret_std": quality["std"],
            })
        scenarios.append({
            "threshold": threshold, "feasible_models": feasible, "optimal_feasible_model": optimal,
            "infeasible": optimal is None,
            "runs_meeting_threshold": sum(run["meets_threshold"] for run in scenario_runs),
            "metrics": {
                field: moments([run[field] for run in scenario_runs]) for field in (
                    "accuracy", "correct", "total_cost_usd", "known_cost_usd", "unknown_cost_count",
                    "fixed_cost_total_usd", "cost_regret_usd", "quality_regret",
                )
            },
        })
    summary = {
        "algorithm": "FT-UCB (Algorithm 6, Appendix H)", "prompts": len(table),
        "models": arm_order, "thresholds": list(thresholds), "repeats": repeats,
        "seeds": list(range(seed, seed + repeats)), "primary_seed": seed,
        "cost_source": "full-data mean of known CSV costs" if empirical_costs else "given fixed costs",
        "arm_costs_usd": costs, "arm_statistics": stats, "scenarios": scenarios,
        "regret_estimator": "full-data plug-in gaps; repeated replays average sampled paths",
    }
    return summary, runs, traces, curves
