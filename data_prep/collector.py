"""Collect real API observations; never invent usage or billed cost."""

import math
import os
import time
from pathlib import Path

import httpx
from filelock import FileLock

from .common import (
    PROMPT_VERSION, SCORING_VERSION, append_attempt, export_results, latest_results,
    load_config, nonnegative_number, read_json, recover_journal, select_models,
    sha256_file, utc_now, write_json,
)
from .dataset import build_messages, load_questions
from .scoring import is_correct, parse_prediction


def api_client(config, require_key=True):
    key = os.environ.get(config["api_key_env"], "").strip()
    if require_key and not key:
        raise ValueError(f"Set the {config['api_key_env']} environment variable before running collection")
    headers = {"X-Title": "LLM_Routing data_prep"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return httpx.Client(
        base_url=config["base_url"].rstrip("/") + "/", headers=headers,
        timeout=config["timeout_seconds"],
    )


def fetch_catalog(client, models):
    response = client.get("models")
    response.raise_for_status()
    entries = response.json().get("data")
    if not isinstance(entries, list):
        raise ValueError("Invalid /models response: expected data list")
    catalog = {entry["id"]: entry for entry in entries if isinstance(entry, dict) and "id" in entry}
    selected = {}
    for model in models:
        identifier = model["id"]
        if identifier not in catalog:
            raise ValueError(f"Model unavailable at this endpoint: {identifier}")
        entry = catalog[identifier]
        if "image" not in entry.get("architecture", {}).get("input_modalities", []):
            raise ValueError(f"Model does not advertise image input: {identifier}")
        selected[identifier] = entry
    return {"fetched_at": utc_now(), "models": selected}


def print_catalog(snapshot):
    print("Live catalog prices in USD per 1M tokens (actual billed cost is recorded separately):")
    for identifier, entry in snapshot["models"].items():
        pricing = entry.get("pricing", {})
        values = [nonnegative_number(pricing.get(key)) for key in ("prompt", "completion")]
        rendered = [f"${value * 1_000_000:.6g}" if value is not None else "unavailable" for value in values]
        print(f"  {identifier}: input {rendered[0]} / output {rendered[1]}")


def lookup_generation(client, generation_id):
    """Read billing metadata, allowing a short delay for eventual consistency."""
    queries = []
    metadata = {}
    for delay in (0, 0.5, 1):
        if delay:
            time.sleep(delay)
        try:
            response = client.get("generation", params={"id": generation_id})
            try:
                payload = response.json()
            except ValueError:
                payload = response.text
            queries.append({"status": response.status_code, "response": payload})
            if response.status_code == 404:
                continue
            response.raise_for_status()
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
                break
            metadata = payload["data"]
            if nonnegative_number(metadata.get("total_cost")) is not None:
                break
        except httpx.HTTPError as exc:
            queries.append({"error": f"{type(exc).__name__}: {exc}"})
            break
    return metadata, queries


def reported_metrics(client, payload):
    usage = payload.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    fields = {
        "input_tokens": ("prompt_tokens", "native_tokens_prompt"),
        "output_tokens": ("completion_tokens", "native_tokens_completion"),
        "cost": ("cost", "total_cost"),
    }
    metrics = {
        name: nonnegative_number(usage.get(keys[0]), integer=name != "cost")
        for name, keys in fields.items()
    }
    sources = {
        name: f"usage.{keys[0]}" if metrics[name] is not None else None
        for name, keys in fields.items()
    }
    metadata, queries = {}, []
    if any(value is None for value in metrics.values()) and isinstance(payload.get("id"), str):
        metadata, queries = lookup_generation(client, payload["id"])
        for name, (_, native) in fields.items():
            if metrics[name] is None:
                metrics[name] = nonnegative_number(metadata.get(native), integer=name != "cost")
                if metrics[name] is not None:
                    sources[name] = f"generation.{native}"
    missing = [name for name, value in metrics.items() if value is None]
    return metrics, sources, metadata, queries, missing


def response_text(payload):
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("API returned no completion choices")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("API returned no assistant message")
    content = message.get("content")
    if content is None and message.get("refusal"):
        return str(message["refusal"]), "Model refused the request"
    if isinstance(content, list):
        content = "".join(
            part["text"] for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    if not isinstance(content, str):
        raise ValueError("API returned no text content")
    reason = choice.get("finish_reason")
    if reason in ("length", "max_tokens", "content_filter", "error"):
        return content, f"Incomplete completion: finish_reason={reason}"
    if choice.get("error"):
        return content, f"API choice error: {choice['error']}"
    return content, None


def collect_one(client, question, model, messages, max_output_tokens):
    row = {
        "question_id": question["question_id"], "subject": question["subject"],
        "model": model["id"], "prediction": None,
        "correct_answer": question["correct_answer"], "correct": 0,
        "input_tokens": None, "output_tokens": None, "cost": None,
        "latency": None, "raw_response": None, "error": None,
    }
    request = {
        **model.get("parameters", {}), "model": model["id"], "messages": messages,
        "max_tokens": max_output_tokens, "stream": False,
    }
    payload = None
    status = None
    errors = []
    started = time.perf_counter()
    try:
        response = client.post("chat/completions", json=request)
        row["latency"] = round(time.perf_counter() - started, 6)
        status = response.status_code
        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        if not isinstance(payload, dict):
            row["raw_response"] = response.text
            errors.append(f"api_error: HTTP {status}; response is not a JSON object")
        elif status >= 400 or payload.get("error"):
            errors.append(f"api_error: HTTP {status}; {payload.get('error', payload)}")
        else:
            try:
                row["raw_response"], completion_error = response_text(payload)
                if completion_error:
                    raise ValueError(completion_error)
                row["prediction"] = parse_prediction(
                    row["raw_response"], question["question_type"], question["options"],
                )
                row["correct"] = is_correct(
                    row["prediction"], question["correct_answer"], question["question_type"],
                )
            except ValueError as exc:
                errors.append(f"parsing_error: {exc}")
    except httpx.HTTPError as exc:
        row["latency"] = round(time.perf_counter() - started, 6)
        errors.append(f"api_error: {type(exc).__name__}: {exc}")

    sources, generation, queries = {}, {}, []
    if isinstance(payload, dict):
        metrics, sources, generation, queries, missing = reported_metrics(client, payload)
        row.update(metrics)
        if missing and not any(error.startswith("api_error:") for error in errors):
            errors.append("accounting_error: API did not report " + ", ".join(missing))
    row["error"] = "; ".join(errors) if errors else None
    return {
        "recorded_at": utc_now(), "result": row, "http_status": status,
        "response": payload, "metric_sources": sources,
        "generation_metadata": generation, "accounting_queries": queries,
    }


def collect_dataset(
    data_dir, output_dir, config_path, model_ids=None, limit=None,
    dry_run=False, retry_errors=False, delay=0,
):
    if not math.isfinite(delay) or delay < 0:
        raise ValueError("delay must be finite and nonnegative")
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    config = load_config(config_path)
    models = select_models(config, model_ids)
    questions = load_questions(data_dir, limit)
    print(f"Selected {len(questions)} questions x {len(models)} models = {len(questions) * len(models)} pairs")
    if dry_run:
        # Exercise the exact prompt path without any HTTP requests or result rows.
        for question in questions:
            build_messages(question, data_dir)
        print("Dry run passed: labels, options, image checksums, and multimodal prompts validated; no API calls")
        return {"requested": 0, "skipped": 0, "errors": 0, "dry_run": True}

    identity = {
        "questions_sha256": sha256_file(data_dir / "questions.jsonl"),
        "base_url": config["base_url"], "max_output_tokens": config["max_output_tokens"],
        "models": config["models"], "prompt_version": PROMPT_VERSION,
        "scoring_version": SCORING_VERSION,
    }
    # Validate credentials before creating any run artifacts.
    with api_client(config) as client:
        output_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(output_dir / "run.lock"), timeout=0):
            manifest_path = output_dir / "run_manifest.json"
            journal_path = output_dir / "attempts.jsonl"
            if manifest_path.exists():
                if read_json(manifest_path)["identity"] != identity:
                    raise ValueError("Run settings or data changed; choose a new --output-dir")
            elif journal_path.exists():
                raise ValueError("Journal exists without run_manifest.json; choose a new --output-dir")
            else:
                write_json(manifest_path, {"created_at": utc_now(), "identity": identity})
            recover_journal(journal_path)
            existing = latest_results(journal_path)
            pending = [
                (question, model) for question in questions for model in models
                if (question["question_id"], model["id"]) not in existing
                or (retry_errors and existing[(question["question_id"], model["id"])]["error"] is not None)
            ]
            skipped = len(questions) * len(models) - len(pending)
            print(f"Pending {len(pending)} pairs; skipping {skipped} previously recorded pairs", flush=True)
            if not pending:
                export_results(output_dir)
                return {"requested": 0, "skipped": skipped, "errors": 0}
            snapshot = fetch_catalog(client, models)
            print_catalog(snapshot)
            snapshot_name = "catalog_" + utc_now().replace(":", "-") + ".json"
            write_json(output_dir / "catalogs" / snapshot_name, snapshot)
            requested = errors = 0
            current_id, messages = None, None
            try:
                for question, model in pending:
                    if requested and delay:
                        time.sleep(delay)
                    if question["question_id"] != current_id:
                        messages = build_messages(question, data_dir)
                        current_id = question["question_id"]
                    attempt = collect_one(client, question, model, messages, config["max_output_tokens"])
                    attempt["catalog_snapshot"] = snapshot_name
                    append_attempt(journal_path, attempt)
                    requested += 1
                    row = attempt["result"]
                    errors += int(row["error"] is not None)
                    state = "error" if row["error"] else f"correct={row['correct']}"
                    print(f"[{requested}/{len(pending)}] {current_id} | {model['id']} | {state} | cost={row['cost']}", flush=True)
                    if attempt["http_status"] in (401, 402, 403):
                        print(f"Stopped on authentication/credit/access error: {row['error']}")
                        break
            finally:
                # Derive exports from the durable journal, also after Ctrl+C.
                if journal_path.exists():
                    export_results(output_dir)
    return {"requested": requested, "skipped": skipped, "errors": errors}
