"""Run with python -m bandit; all inputs are local and no model APIs are called."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from .data import SCORER, load_data
from .evaluate import evaluate


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
                for key, value in row.items()
            })


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_report(summary):
    lines = [
        "# Offline UCB1 routing experiment", "",
        f"Input: `{summary['input_csv']}`", "",
        f"{summary['prompts']} prompts x {len(summary['models'])} models; "
        f"{summary['repeats']} shuffled replays, seeds {summary['seeds'][0]} through {summary['seeds'][-1]}.", "",
        "Reward: correct = 1, incorrect / empty / failed request = 0. "
        "Each replay starts with an empty UCB1 state and uses each prompt once. "
        "Only the selected model's reward updates the policy. Exploration rounds count toward all metrics.", "",
        *cost_reward_note(summary["values_per_correct"]),
        "| Policy | Accuracy (mean +/- SD) | Known cost, USD (mean) | Unknown costs (mean count) | Regret vs best fixed (mean) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for policy, values in summary["policies"].items():
        accuracy = values["accuracy"]
        lines.append(
            f"| {policy} | {accuracy['mean']:.2%} +/- {accuracy['std']:.2%} "
            f"| {values['known_cost_usd']['mean']:.6f} "
            f"| {values['unknown_cost_count']['mean']:.2f} "
            f"| {values['regret_vs_best_fixed']['mean']:.2f} |"
        )
    lines.extend([
        "", f"Oracle accuracy upper bound: {summary['oracle_accuracy']:.2%}. "
        "The oracle sees every model's score; it is only a comparison and cannot be deployed.", "",
        "Regret is the best fixed model's number of correct answers minus the policy's. "
        "In the trace, the comparator is recomputed for each prefix. It may be negative when a policy "
        "outperforms every fixed model. The oracle gap is a separate, nonnegative quantity.", "",
        "Cost covers only selected CSV observations. Missing costs remain unknown and are never replaced "
        "with zero; reported known cost is a lower bound when unknown entries are selected. "
        "Historical retries and waiting time are not fully represented by this CSV.", "",
        f"Scorer: {summary['scorer']}. "
        f"{summary['grading']['needs_review']} automatically scored observations are flagged for review "
        "in scored_results.csv, including all open questions without reviewed labels. "
        "These automatic scores are provisional; review them before reporting final benchmark results.", "",
        "The SD describes sensitivity to prompt order and random choices on this same dataset. "
        "It is not a confidence interval for unseen prompts. This is an online-learning replay "
        "with immediate ground-truth feedback, and uses no separate training/test split. "
        "UCB1 ignores prompt features. Plain ucb1 optimizes accuracy only. "
        "Latency is reported separately and never enters a reward.", "",
    ])
    return "\n".join(lines)


def cost_reward_note(values_per_correct):
    if not values_per_correct:
        return []
    return [
        "Each ucb1-cost[V=v] policy is UCB1 with reward correct - min(cost / v, 1), "
        "mapped from [-1, 1] to [0, 1]. Here v is the USD value of one correct answer. "
        "An unknown cost gets the full penalty of 1.", "",
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline UCB1 routing from an existing complete response CSV.")
    parser.add_argument("--input", type=Path, default=Path("data_prep/output/run/result.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("bandit/output/ucb1"))
    parser.add_argument("--expected-prompts", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--value-per-correct", type=float, nargs="+", default=[], metavar="USD",
        help="Add one cost-aware UCB1 policy for each value: the USD value of one correct answer.",
    )
    parser.add_argument("--labels", type=Path, help="Optional reviewed question_id,model,correct CSV (0/1).")
    args = parser.parse_args(argv)
    if args.expected_prompts < 1 or args.repeats < 1:
        parser.error("--expected-prompts and --repeats must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in args.value_per_correct):
        parser.error("--value-per-correct values must be positive")
    try:
        table, models, scored = load_data(args.input, args.expected_prompts, args.labels)
        summary, runs, traces = evaluate(
            table, models, args.repeats, args.seed, args.value_per_correct,
        )
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    summary.update(
        input_csv=str(args.input.resolve()), input_sha256=file_hash(args.input),
        scorer=SCORER,
        labels_csv=str(args.labels.resolve()) if args.labels else None,
        labels_sha256=file_hash(args.labels) if args.labels else None,
        grading={
            "observations": len(scored),
            "needs_review": sum(row["needs_review"] for row in scored),
            "score_sources": dict(Counter(row["score_source"] for row in scored)),
            "extraction_counts": dict(Counter(row["extraction"] for row in scored)),
            "question_types": dict(Counter(row["question_type"] for row in scored)),
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "scored_results.csv", scored)
    write_csv(args.output_dir / "runs.csv", runs)
    write_csv(args.output_dir / "trace.csv", traces)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    report = make_report(summary)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
