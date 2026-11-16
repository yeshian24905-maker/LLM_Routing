"""Small paid trial on one model, followed by a human-readable report for manual review."""

import re
from pathlib import Path

from .collector import api_client, catalog_price, collect_dataset, fetch_catalog
from .common import latest_results, load_config, read_json, summarize_rows, utc_now
from .dataset import load_questions


def cheapest_model(catalog):
    """Lowest listed input + output price per token among models with known prices."""
    priced = {
        identifier: catalog_price(entry, "prompt") + catalog_price(entry, "completion")
        for identifier, entry in catalog["models"].items()
        if catalog_price(entry, "prompt") is not None and catalog_price(entry, "completion") is not None
    }
    if not priced:
        raise ValueError("No model in the catalog lists prices; pass --model explicitly")
    return min(priced, key=priced.get)


def fenced(text):
    """Markdown code fence that cannot be closed early by the response itself."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def number(value, digits=1):
    return "n/a" if value is None else f"{value:,.{digits}f}"


def write_report(run_dir, model_id, catalog, total_questions):
    run_dir = Path(run_dir)
    rows = [row for row in latest_results(run_dir / "attempts.jsonl").values() if row["model"] == model_id]
    rows.sort(key=lambda row: row["question_id"])
    stats = summarize_rows(rows)
    tagged = sum(bool(re.search(r"<answer>.*?</answer>", row["raw_response"] or "", re.S)) for row in rows)
    priced_rows = [row for row in rows if row["cost"] is not None]
    mean_cost = stats["known_cost_usd"] / len(priced_rows) if priced_rows else None

    lines = [
        f"# Pilot report: {model_id}", "",
        f"Generated {utc_now()} from `{run_dir}`. Responses are shown as recorded; nothing is scored.", "",
        "## Overview", "",
        "| Metric | Value |", "| --- | --- |",
        f"| Questions | {len(rows)} |",
        f"| API errors | {stats['api_errors']} |",
        f"| Other errors (e.g. missing billing data) | {stats['errors'] - stats['api_errors']} |",
        f"| finish_reason | {', '.join(f'{key}: {value}' for key, value in sorted(stats['finish_reasons'].items()))} |",
        f"| Empty responses | {stats['empty_responses']} |",
        f"| Responses containing an `<answer>...</answer>` tag | {tagged} / {len(rows)} |",
        f"| Mean input / output / reasoning tokens | {number(stats['mean_input_tokens'])} / "
        f"{number(stats['mean_output_tokens'])} / {number(stats['mean_reasoning_tokens'])} |",
        f"| Latency p50 / p95 (s) | {number(stats['p50_latency_seconds'], 2)} / {number(stats['p95_latency_seconds'], 2)} |",
        f"| Known cost | ${stats['known_cost_usd']:.4f} ({stats['rows_with_unknown_cost']} rows without cost) |",
        "",
        "## Cost projection", "",
        f"Mean actual cost per request for {model_id}: "
        + (f"${mean_cost:.6f}." if mean_cost is not None else "unknown."),
        "",
        f"Other models are estimated from live catalog prices and **this pilot's mean token counts**. "
        "Real output and reasoning lengths differ between models, so treat these as rough lower/upper hints, "
        "not quotes.",
        "",
        f"| Model | Estimated cost per request | Estimated cost for {total_questions} questions |",
        "| --- | --- | --- |",
    ]
    for identifier, entry in catalog["models"].items():
        if identifier == model_id and mean_cost is not None:
            per_request = mean_cost
        else:
            prompt, completion = catalog_price(entry, "prompt"), catalog_price(entry, "completion")
            usable = None not in (prompt, completion, stats["mean_input_tokens"], stats["mean_output_tokens"])
            per_request = (
                prompt * stats["mean_input_tokens"] + completion * stats["mean_output_tokens"]
                if usable else None
            )
        label = f"{identifier} (measured)" if identifier == model_id else identifier
        if per_request is None:
            lines.append(f"| {label} | unknown | unknown |")
        else:
            lines.append(f"| {label} | ${per_request:.6f} | ${per_request * total_questions:.2f} |")
    lines += ["", "## Responses", ""]
    for row in rows:
        lines += [
            f"### {row['question_id']} ({row['subject']}, {row['question_type']})", "",
            f"- Correct answer: `{row['correct_answer']}`",
            f"- finish_reason: `{row['finish_reason']}`; tokens in/out/reasoning: "
            f"{row['input_tokens']}/{row['output_tokens']}/{row['reasoning_tokens']}; "
            f"cost: {row['cost']}; latency: {row['latency']} s",
        ]
        if row["error"]:
            lines.append(f"- Error: {row['error']}")
        lines += ["", fenced(row["raw_response"] if row["raw_response"] is not None else "(no response text)"), ""]
    path = run_dir / "pilot_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path, stats


def run_pilot(data_dir, config_path, questions=10, model_id=None, output_dir=None, workers_per_model=None):
    data_dir = Path(data_dir)
    config = load_config(config_path)
    load_questions(data_dir, questions)  # Fail locally before any network access.
    with api_client(config) as client:
        catalog = fetch_catalog(client, config["models"], config)
    model_id = model_id or cheapest_model(catalog)
    if model_id not in catalog["models"]:
        raise ValueError(f"Model not in config: {model_id}")
    print(f"Pilot model: {model_id}", flush=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model_id).strip("-")
    output_dir = Path(output_dir) if output_dir else data_dir / "runs" / f"pilot_{slug}_{questions}"
    summary = collect_dataset(
        data_dir, output_dir, config_path, model_ids=[model_id], limit=questions,
        workers_per_model=workers_per_model,
    )
    manifest = data_dir / "dataset_manifest.json"
    total = read_json(manifest)["question_count"] if manifest.exists() else len(load_questions(data_dir))
    path, stats = write_report(output_dir, model_id, catalog, total)
    print(f"Pilot report written to {path}")
    print(
        f"  {stats['rows']} responses, {stats['api_errors']} API errors, finish_reason {stats['finish_reasons']}, "
        f"known cost ${stats['known_cost_usd']:.4f}"
    )
    return summary, path
