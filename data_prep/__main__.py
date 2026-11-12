"""Command line entry point: python -m data_prep --help."""

import argparse
import sys
from pathlib import Path

from filelock import FileLock

from .collector import api_client, collect_dataset, fetch_catalog, print_catalog
from .common import (
    DEFAULT_CONFIG, DEFAULT_DATA_DIR, export_results, load_config, recover_journal,
    select_models, write_json,
)
from .dataset import download_dataset


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parser():
    root = argparse.ArgumentParser(description="Prepare real multimodal MMMU model-routing observations")
    root.add_argument("--debug", action="store_true", help="Show a traceback on failure")
    commands = root.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download labeled MMMU questions and images")
    download.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    download.add_argument("--splits", nargs="+", choices=("dev", "validation", "test"), default=["validation"])
    download.add_argument("--subjects", nargs="+", help="Exact Hugging Face subject/config names; default: all 30")
    download.add_argument("--limit-per-subject", type=positive_int, help="Maximum questions per subject per split (in source order)")
    download.add_argument("--revision", default="main", help="Dataset branch, tag, or commit; resolved SHA is recorded")

    check = commands.add_parser("check-models", help="Check model IDs, image support, and live catalog prices")
    check.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    check.add_argument("--models", nargs="+")
    check.add_argument("--output", type=Path, help="Optionally save the current selected catalog entries")

    run = commands.add_parser("run", help="Query each selected question/model pair and save observations")
    run.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    run.add_argument("--output-dir", type=Path, help="Default: DATA_DIR/runs/default")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--models", nargs="+", help="Subset of model IDs in config.json")
    run.add_argument("--limit", type=positive_int, help="Use the first N prepared questions")
    run.add_argument("--dry-run", action="store_true", help="Validate inputs and prompts offline; make no API calls")
    run.add_argument("--retry-errors", action="store_true", help="Re-query recorded error rows (creates new billable attempts)")
    run.add_argument("--delay", type=float, default=0, help="Seconds between completion requests")

    export = commands.add_parser("export", help="Rebuild CSV, JSONL, and summary from the attempts journal")
    export.add_argument("--run-dir", type=Path, default=DEFAULT_DATA_DIR / "runs" / "default")
    return root


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "download":
            download_dataset(
                arguments.output_dir, arguments.splits, arguments.subjects,
                arguments.limit_per_subject, arguments.revision,
            )
        elif arguments.command == "check-models":
            config = load_config(arguments.config)
            with api_client(config, require_key=False) as client:
                snapshot = fetch_catalog(client, select_models(config, arguments.models))
            print_catalog(snapshot)
            if arguments.output:
                write_json(arguments.output, snapshot)
        elif arguments.command == "run":
            summary = collect_dataset(
                data_dir=arguments.data_dir,
                output_dir=arguments.output_dir or arguments.data_dir / "runs" / "default",
                config_path=arguments.config, model_ids=arguments.models, limit=arguments.limit,
                dry_run=arguments.dry_run, retry_errors=arguments.retry_errors, delay=arguments.delay,
            )
            print(f"Collection finished: {summary}")
            return int(summary["errors"] > 0)
        elif arguments.command == "export":
            with FileLock(str(arguments.run_dir / "run.lock"), timeout=0):
                recover_journal(arguments.run_dir / "attempts.jsonl")
                export_results(arguments.run_dir)
    except KeyboardInterrupt:
        print("\nInterrupted; recorded attempts are preserved. Run the same command to resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        if arguments.debug:
            raise
        print(f"Error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
