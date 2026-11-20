"""Route the MMMU prompt list with FT-UCB, reusing its saved model outcomes."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .data import SCORER, load_data, load_questions
from .ft_evaluate import DEFAULT_THRESHOLDS, evaluate_ft
from .ft_report import make_ft_report, plot_ft


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline FT-UCB routing (paper Algorithm 6).")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "mmmu_test_sample",
                        help="MMMU directory containing questions.jsonl, images and runs.")
    parser.add_argument("--run", default="sequential", help="Saved run name under DATA_DIR/runs.")
    parser.add_argument("--output-dir", type=Path, default=Path("bandit/output/ft-ucb"))
    parser.add_argument("--expected-prompts", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels", type=Path, help="Reviewed question_id,model,correct CSV (0/1).")
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS,
                        help="Default: 0.89 0.85 0.80 0.825.")
    costs = parser.add_mutually_exclusive_group(required=True)
    costs.add_argument("--arm-costs", type=Path, help="JSON mapping each model to fixed USD/call cost.")
    costs.add_argument("--empirical-costs", action="store_true",
                       help="Approximate fixed costs by full-data mean known CSV costs.")
    args = parser.parse_args(argv)
    if args.expected_prompts < 1 or args.repeats < 1:
        parser.error("--expected-prompts and --repeats must be positive")
    try:
        questions = load_questions(args.data_dir, args.expected_prompts)
        runs_dir = (args.data_dir / "runs").resolve()
        run_dir = (runs_dir / args.run).resolve()
        if not run_dir.is_relative_to(runs_dir):
            raise ValueError("--run must select a directory inside DATA_DIR/runs")
        questions_path = args.data_dir / "questions.jsonl"
        run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8-sig"))
        if run_manifest.get("identity", {}).get("questions_sha256") != file_hash(questions_path):
            raise ValueError("Saved model run was collected on a different questions.jsonl")
        results_path = run_dir / "results.csv"
        table, models, scored = load_data(results_path, questions, args.labels)
        fixed_costs = json.loads(args.arm_costs.read_text(encoding="utf-8-sig")) if args.arm_costs else None
        summary, runs, traces, curves = evaluate_ft(
            table, models, args.thresholds, args.repeats, args.seed, fixed_costs, args.empirical_costs,
        )
        summary.update(
            questions_jsonl=str(questions_path.resolve()), questions_sha256=file_hash(questions_path),
            input_csv=str(results_path), input_sha256=file_hash(results_path),
            arm_costs_json=str(args.arm_costs.resolve()) if args.arm_costs else None,
            arm_costs_sha256=file_hash(args.arm_costs) if args.arm_costs else None,
            labels_csv=str(args.labels.resolve()) if args.labels else None,
            labels_sha256=file_hash(args.labels) if args.labels else None,
            scorer=SCORER, grading={"needs_review": sum(row["needs_review"] for row in scored)},
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot_ft(args.output_dir, summary, runs, curves)
        for name, rows in (("scored_results", scored), ("runs", runs), ("trace", traces), ("curves", curves)):
            write_csv(args.output_dir / f"{name}.csv", rows)
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        report = make_ft_report(summary, runs)
        (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    except ImportError as exc:
        parser.error(f"Install plotting dependencies: python -m pip install -r bandit/requirements.txt ({exc})")
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    print(report)
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
