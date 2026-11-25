"""Paired, network-disabled comparison of the requested confidence policies."""

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import socket
import statistics

from .confidence import FixedThresholdConfidence, POLICIES
from .data import load_csv_replay
from .ft_evaluate import DEFAULT_THRESHOLDS, arm_statistics, resolve_costs


@contextmanager
def network_disabled():
    """Fail any attempted socket connection during the complete experiment."""
    attempts = []
    originals = socket.create_connection, socket.socket.connect, socket.socket.connect_ex

    def deny(*args, **kwargs):
        attempts.append("connection blocked")
        raise RuntimeError("Network connections are disabled for offline comparisons")

    socket.create_connection = socket.socket.connect = socket.socket.connect_ex = deny
    try:
        yield attempts
    finally:
        socket.create_connection, socket.socket.connect, socket.socket.connect_ex = originals


def prompt_stream(question_ids, horizon, seed):
    """One shuffled pass, or explicitly resampled rows for a longer horizon."""
    if not question_ids or horizon < 1:
        raise ValueError("Need question IDs and a positive horizon")
    ids = list(question_ids)
    if horizon <= len(ids):
        random.Random(f"order:{seed}").shuffle(ids)
        return ids[:horizon], "without_replacement"
    return random.Random(f"bootstrap:{seed}").choices(ids, k=horizon), "with_replacement"


def _moments(values):
    if any(value is None for value in values):
        return {"mean": None, "std": None}
    return {"mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0}


def summarize_runs(runs, policy_ids, horizons, thresholds):
    rows = []
    for horizon in horizons:
        for threshold in thresholds:
            case = [r for r in runs if r["horizon"] == horizon and r["threshold"] == threshold]
            baseline = {r["seed"]: r for r in case if r["policy"] == "ucb_original"}
            for policy in policy_ids:
                group = [r for r in case if r["policy"] == policy]
                item = {"horizon": horizon, "sampling": group[0]["sampling"],
                        "threshold": threshold, "policy": policy, "repeats": len(group)}
                for metric in ("accuracy", "known_cost_usd", "total_cost_usd", "fixed_cost_usd",
                               "cost_regret_usd", "quality_regret", "best_arm_regret", "fallback_fraction"):
                    values = _moments([r[metric] for r in group])
                    item[metric + "_mean"] = values["mean"]
                    item[metric + "_std"] = values["std"]
                deltas = [(r["accuracy"]-baseline[r["seed"]]["accuracy"])*100 for r in group]
                item["paired_accuracy_gain_pp"] = statistics.mean(deltas)
                sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
                half = 1.95996398454 * sd / math.sqrt(len(deltas)) if len(deltas) > 1 else None
                item["gain_mc_ci95_low_pp"] = item["paired_accuracy_gain_pp"] - half if half is not None else None
                item["gain_mc_ci95_high_pp"] = item["paired_accuracy_gain_pp"] + half if half is not None else None
                rows.append(item)
    return rows


TRACE_FIELDS = ("horizon", "sampling", "policy", "threshold", "seed", "step", "question_id",
    "model", "phase", "prior_arm_count", "prior_arm_mean", "selected_bound", "candidate_models",
    "correct", "cost_usd", "cumulative_correct", "cumulative_accuracy", "known_cost_usd",
    "unknown_cost_count", "fixed_cost_usd", "cost_regret_usd", "quality_regret", "best_arm_regret")


def run_comparison(table, models, costs, *, policy_ids=tuple(POLICIES), horizons=None,
                   thresholds=DEFAULT_THRESHOLDS, repeats=30, seed=42, trace_writer=None,
                   progress=None):
    horizons = tuple(horizons or (len(table), 10000))
    thresholds = tuple(thresholds)
    policy_ids = tuple(policy_ids)
    if len(models) != 4 or repeats < 1 or not horizons or len(set(horizons)) != len(horizons) or any(h < 1 for h in horizons):
        raise ValueError("Need four models, positive repeats and distinct positive horizons")
    if not policy_ids or len(set(policy_ids)) != len(policy_ids) or any(p not in POLICIES for p in policy_ids) or "ucb_original" not in policy_ids:
        raise ValueError("Use distinct known policies including ucb_original as the paired baseline")
    if not thresholds or len(set(thresholds)) != len(thresholds) or any(not math.isfinite(t) or not 0 < t <= 1 for t in thresholds):
        raise ValueError("Use distinct thresholds in (0, 1]")
    stats = arm_statistics(table, models)
    means = {m: stats[m]["accuracy"] for m in models}
    best_mean = max(means.values())
    arm_order = sorted(models, key=lambda m: (costs[m], m))
    runs, checkpoints = [], {}
    for horizon in horizons:
        for run_seed in range(seed, seed+repeats):
            stream, sampling = prompt_stream(table, horizon, run_seed)
            for threshold in thresholds:
                feasible = [m for m in arm_order if means[m] >= threshold]
                reference = feasible[0] if feasible else None
                for policy_id in policy_ids:
                    policy = FixedThresholdConfidence(models, costs, threshold, policy_id, seed=f"ft:{run_seed}")
                    correct, known, missing, fixed, quality, reward_regret, fallback = 0, 0.0, 0, 0.0, 0.0, 0.0, 0
                    cost_regret = 0.0 if reference is not None else None
                    save_trace = run_seed == seed and trace_writer is not None
                    for t, question_id in enumerate(stream, 1):
                        arm = policy.select(diagnostics=save_trace)
                        count = policy.counts[arm]
                        prior = policy.successes[arm]/count if count else None
                        # No unsampled outcome or full-data mean is passed to the policy.
                        observation = table[question_id][arm]
                        policy.update(arm, observation.correct)
                        correct += observation.correct
                        if observation.cost is None:
                            missing += 1
                        else:
                            known += observation.cost
                        fixed += costs[arm]
                        quality += max(0.0, threshold-means[arm])
                        reward_regret += best_mean-means[arm]
                        if reference is not None:
                            cost_regret += max(0.0, costs[arm]-costs[reference])
                        fallback += policy.last_decision["phase"] == "uniform_fallback"
                        if save_trace:
                            trace_writer.writerow({
                                "horizon": horizon, "sampling": sampling, "policy": policy_id,
                                "threshold": threshold, "seed": run_seed, "step": t, "question_id": question_id,
                                "model": arm, "phase": policy.last_decision["phase"], "prior_arm_count": count,
                                "prior_arm_mean": prior, "selected_bound": policy.last_decision["selected_bound"],
                                "candidate_models": json.dumps(policy.last_decision["candidates"]),
                                "correct": observation.correct, "cost_usd": observation.cost,
                                "cumulative_correct": correct, "cumulative_accuracy": correct/t,
                                "known_cost_usd": known, "unknown_cost_count": missing, "fixed_cost_usd": fixed,
                                "cost_regret_usd": cost_regret, "quality_regret": quality,
                                "best_arm_regret": reward_regret,
                            })
                        if t <= 4 or t % 50 == 0 or t == horizon:
                            key = horizon, sampling, threshold, policy_id, t
                            checkpoints.setdefault(key, []).append((correct/t, fixed, cost_regret, quality))
                    runs.append({"horizon": horizon, "sampling": sampling, "unique_questions": len(set(stream)),
                        "policy": policy_id, "threshold": threshold, "seed": run_seed, "correct": correct,
                        "accuracy": correct/horizon, "known_cost_usd": known, "unknown_cost_count": missing,
                        "total_cost_usd": known if not missing else None, "fixed_cost_usd": fixed,
                        "cost_regret_usd": cost_regret, "quality_regret": quality,
                        "best_arm_regret": reward_regret, "fallback_count": fallback,
                        "fallback_fraction": fallback/horizon, "model_selections": dict(policy.counts)})
            if progress:
                progress(f"Completed horizon={horizon}, seed={run_seed} ({run_seed-seed+1}/{repeats})")
    curves = []
    for (horizon, sampling, threshold, policy_id, t), values in checkpoints.items():
        row = {"horizon": horizon, "sampling": sampling, "threshold": threshold, "policy": policy_id, "step": t}
        for j, metric in enumerate(("accuracy", "fixed_cost_usd", "cost_regret_usd", "quality_regret")):
            for name, value in _moments([v[j] for v in values]).items():
                row[metric + "_" + name] = value
        curves.append(row)
    return runs, summarize_runs(runs, policy_ids, horizons, thresholds), curves


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, default=root/"results_3000.csv")
    parser.add_argument("--scores-csv", type=Path, default=root/"bandit/output/ft-ucb/scored_results.csv")
    parser.add_argument("--expected-prompts", type=int, default=3000)
    parser.add_argument("--horizons", type=int, nargs="+", default=[3000, 10000])
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=root/"bandit/output/confidence-comparison")
    parser.add_argument("--pdf", type=Path, default=root/"artifacts/Confidence_Methods_Comparison.pdf")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--empirical-costs", action="store_true")
    source.add_argument("--arm-costs", type=Path)
    args = parser.parse_args(argv)
    with network_disabled() as attempts:
        source_hash = hashlib.sha256(args.results_csv.read_bytes()).hexdigest()
        archive_hash = hashlib.sha256(args.scores_csv.read_bytes()).hexdigest()
        table, models, scored = load_csv_replay(args.results_csv, args.scores_csv, args.expected_prompts)
        stats = arm_statistics(table, models)
        fixed = json.loads(args.arm_costs.read_text(encoding="utf-8-sig")) if args.arm_costs else None
        costs = resolve_costs(models, stats, fixed, args.empirical_costs)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir/"primary_trace.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS)
            writer.writeheader()
            runs, aggregates, curves = run_comparison(table, models, costs,
                policy_ids=args.policies, horizons=args.horizons, thresholds=args.thresholds,
                repeats=args.repeats, seed=args.seed, trace_writer=writer, progress=lambda s: print(s, flush=True))
        assert source_hash == hashlib.sha256(args.results_csv.read_bytes()).hexdigest()
        assert archive_hash == hashlib.sha256(args.scores_csv.read_bytes()).hexdigest()
        summary = {"questions": len(table), "outcomes": len(scored), "policies": {p: POLICIES[p] for p in args.policies},
            "horizons": args.horizons, "thresholds": args.thresholds, "repeats": args.repeats,
            "primary_seed": args.seed, "arm_costs_usd": costs, "arm_statistics": stats,
            "cost_source": "full-data mean of known costs" if args.empirical_costs else "supplied fixed costs",
            "input_csv": str(args.results_csv), "input_sha256": source_hash,
            "archive_csv": str(args.scores_csv), "archive_sha256": archive_hash,
            "needs_review": sum(int(row["needs_review"]) for row in scored),
            "network_guard_enabled": True, "network_connections_attempted": len(attempts),
            "decisions": sum(r["horizon"] for r in runs), "runs": len(runs),
            "sampling_note": "3000 rounds: one shuffled pass. 10000 rounds: sampling the same pool with replacement; no new questions.",
            "uncertainty_note": "SD and normal-approximation paired CIs measure order/resampling sensitivity conditional on fixed labels, not unseen-data uncertainty."}
        for name, rows in (("runs", runs), ("aggregates", aggregates), ("curves", curves)):
            write_csv(args.output_dir/f"{name}.csv", rows)
        (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
        from .comparison_report import render_comparison
        render_comparison(summary, runs, aggregates, curves, args.pdf, args.output_dir)
    print(f"Complete: {summary['runs']} runs, {summary['decisions']:,} decisions; network attempts: {len(attempts)}")
    print(f"PDF: {args.pdf}")


if __name__ == "__main__":
    main()
