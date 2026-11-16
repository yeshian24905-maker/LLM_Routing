"""File formats and configuration shared by the preparation commands."""

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "config.json"
DEFAULT_DATA_DIR = PACKAGE_DIR / "output" / "mmmu_test_sample"
FIELDS = (
    "question_id", "subject", "question_type", "model", "raw_response", "correct_answer",
    "finish_reason", "input_tokens", "output_tokens", "reasoning_tokens", "cost", "latency", "error",
)
PROMPT_VERSION = "mmmu-final-answer-v1"
# Request keys the collector always sets itself; config must not override them.
RESERVED_REQUEST_KEYS = {"model", "models", "messages", "prompt", "stream", "max_tokens"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json_dumps(row) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield row


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path=DEFAULT_CONFIG):
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    for name in ("base_url", "api_key_env"):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ValueError(f"Config requires {name}")
    endpoint = urlsplit(config["base_url"])
    local_http = endpoint.scheme == "http" and endpoint.hostname in ("localhost", "127.0.0.1", "::1")
    if not endpoint.hostname or (endpoint.scheme != "https" and not local_http):
        raise ValueError("base_url must use HTTPS (HTTP is allowed for local testing)")
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError("base_url must not contain credentials, a query, or a fragment")
    for name in ("timeout_seconds", "max_output_tokens", "max_attempts", "concurrency_per_model"):
        value = config.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Config {name} must be a positive integer")
    shared = config.setdefault("request_parameters", {})
    if not isinstance(shared, dict) or RESERVED_REQUEST_KEYS.intersection(shared):
        raise ValueError("Invalid request_parameters in config")
    if not isinstance(config.get("models"), list) or not config["models"]:
        raise ValueError("Config requires at least one model")
    identifiers = []
    for model in config["models"]:
        if not isinstance(model, dict) or not isinstance(model.get("id"), str) or not model["id"]:
            raise ValueError("Each model requires an id")
        parameters = model.get("parameters", {})
        if not isinstance(parameters, dict) or RESERVED_REQUEST_KEYS.intersection(parameters):
            raise ValueError(f"Invalid additional parameters for {model['id']}")
        identifiers.append(model["id"])
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate model IDs in config")
    return config


def select_models(config, identifiers=None):
    models = config["models"]
    if identifiers:
        missing = set(identifiers) - {model["id"] for model in models}
        if missing:
            raise ValueError(f"Models missing from config: {sorted(missing)}")
        models = [model for model in models if model["id"] in identifiers]
    return models


def request_parameters(config, model):
    """Shared request settings, overridden by the model's own parameters."""
    return {**config.get("request_parameters", {}), **model.get("parameters", {})}


def nonnegative_number(value, integer=False):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    if integer:
        return int(number) if number.is_integer() else None
    return number


def percentile(values, fraction):
    """Nearest-rank percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def append_attempt(path, attempt):
    # One durable record contains both the observation and its original API payload.
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json_dumps(attempt) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def recover_journal(path):
    """Preserve and remove an incomplete final write before appending on resume."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r+b") as stream:
        line_number = 0
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                return
            line_number += 1
            try:
                json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if line.endswith(b"\n") or stream.read(1):
                    raise ValueError(f"Corrupt journal {path}:{line_number}") from exc
                backup = path.with_name(path.name + ".partial")
                with backup.open("ab") as saved:
                    saved.write(line + b"\n")
                stream.seek(offset)
                stream.truncate()
                print(f"Recovered incomplete journal tail; saved to {backup}")
                return
            if not line.endswith(b"\n"):
                stream.write(b"\n")
                return


def latest_records(journal_path):
    """Latest journal record per (question_id, model) pair."""
    latest = {}
    if Path(journal_path).exists():
        for record in read_jsonl(journal_path):
            row = record["result"]
            if set(row) != set(FIELDS):
                raise ValueError("Unexpected result schema in attempts.jsonl")
            latest[(row["question_id"], row["model"])] = record
    return latest


def latest_results(journal_path):
    return {key: record["result"] for key, record in latest_records(journal_path).items()}


def is_api_error(row):
    return row["error"] is not None and row["error"].startswith("api_error")


def request_costs(journal_path):
    """Billing per completion request, counting accounting refreshes only once."""
    costs, failed = {}, 0
    for number, record in enumerate(read_jsonl(journal_path)):
        if record.get("kind") == "accounting_refresh":
            costs[record["generation_id"]] = record["result"]["cost"]
            continue
        response = record.get("response")
        generation_id = response.get("id") if isinstance(response, dict) else None
        if is_api_error(record["result"]) and record["result"]["cost"] is None:
            failed += 1
            continue
        costs[generation_id if isinstance(generation_id, str) else f"line-{number}"] = record["result"]["cost"]
    known = [cost for cost in costs.values() if cost is not None]
    return {
        "completion_requests": len(costs), "known_cost_usd": math.fsum(known),
        "requests_with_unknown_cost": len(costs) - len(known), "failed_requests": failed,
    }


def summarize_rows(rows):
    def mean(values):
        return sum(values) / len(values) if values else None

    def present(name):
        return [row[name] for row in rows if row[name] is not None]

    latencies = present("latency")
    costs = present("cost")
    return {
        "rows": len(rows),
        "errors": sum(row["error"] is not None for row in rows),
        "api_errors": sum(is_api_error(row) for row in rows),
        "finish_reasons": dict(Counter(str(row["finish_reason"]) for row in rows)),
        "empty_responses": sum(not (row["raw_response"] or "").strip() for row in rows),
        "mean_input_tokens": mean(present("input_tokens")),
        "mean_output_tokens": mean(present("output_tokens")),
        "mean_reasoning_tokens": mean(present("reasoning_tokens")),
        "known_cost_usd": math.fsum(costs), "rows_with_unknown_cost": len(rows) - len(costs),
        "mean_latency_seconds": mean(latencies),
        "p50_latency_seconds": percentile(latencies, 0.5),
        "p95_latency_seconds": percentile(latencies, 0.95),
    }


def export_results(run_dir):
    """Export the latest observation per pair; billing totals cover every request."""
    run_dir = Path(run_dir)
    journal = run_dir / "attempts.jsonl"
    if not journal.exists():
        raise ValueError(f"No attempts journal at {journal}")
    latest = latest_results(journal)
    rows = [latest[key] for key in sorted(latest)]
    write_jsonl(run_dir / "results.jsonl", rows)
    temporary = run_dir / "results.csv.tmp"
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json_dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })
    temporary.replace(run_dir / "results.csv")
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"]].append(row)
    summary = {
        "exported_at": utc_now(), "rows": len(rows),
        "models": {model: summarize_rows(observations) for model, observations in sorted(groups.items())},
        "all_requests": request_costs(journal),
    }
    write_json(run_dir / "summary.json", summary)
    print(f"Exported {len(rows)} rows to {run_dir / 'results.csv'} and results.jsonl")
    return summary
