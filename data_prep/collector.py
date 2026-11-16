"""Collect one real API observation per question/model pair; never invent usage or cost.

Responses are recorded exactly as returned: nothing is parsed, scored, or re-asked here.
Only requests that produced no model response (network failures, 408/429/5xx) are retried.
"""

import os
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx
from filelock import FileLock

from .common import (
    FIELDS, PROMPT_VERSION, append_attempt, export_results, is_api_error, latest_records,
    load_config, nonnegative_number, read_json, recover_journal, request_parameters,
    select_models, sha256_file, utc_now, write_json,
)
from .dataset import build_messages, load_questions


RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 529}
STOP_STATUS = {401, 402, 403}
GENERATION_LOOKUP_DELAYS = (0, 1, 2, 4, 8)
MAX_RETRY_DELAY = 120


class RequestPacer:
    """Space every POST attempt across all workers serving the same model."""

    def __init__(self, requests_per_minute):
        self.interval = 60.0 / requests_per_minute
        self.next_allowed = 0.0
        self.lock = threading.Lock()

    def wait(self, stop=None):
        while stop is None or not stop.is_set():
            with self.lock:
                now = time.monotonic()
                delay = self.next_allowed - now
                if delay <= 0:
                    self.next_allowed = now + self.interval
                    return True
            if stop is None:
                time.sleep(delay)
            elif stop.wait(delay):
                return False
        return False

    def defer(self, seconds):
        with self.lock:
            self.next_allowed = max(self.next_allowed, time.monotonic() + seconds)


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


def fetch_catalog(client, models, config=None):
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
        supported = entry.get("supported_parameters")
        if config is not None and isinstance(supported, list):
            unsupported = sorted(set(request_parameters(config, model)) - set(supported) - {"usage"})
            if unsupported:
                raise ValueError(f"{identifier} does not list support for request parameters {unsupported}")
        selected[identifier] = entry
    return {"fetched_at": utc_now(), "models": selected}


def catalog_price(entry, key):
    return nonnegative_number(entry.get("pricing", {}).get(key))


def print_catalog(snapshot):
    print("Live catalog prices in USD per 1M tokens (actual billed cost is recorded separately):")
    for identifier, entry in snapshot["models"].items():
        values = [catalog_price(entry, key) for key in ("prompt", "completion")]
        rendered = [f"${value * 1_000_000:.6g}" if value is not None else "unavailable" for value in values]
        print(f"  {identifier}: input {rendered[0]} / output {rendered[1]}")


def lookup_generation(client, generation_id):
    """Read billing metadata, allowing time for eventual consistency (about 15 s at most)."""
    queries = []
    metadata = {}
    for delay in GENERATION_LOOKUP_DELAYS:
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
    """Token counts and billed cost as reported; missing reasoning tokens are not an error."""
    usage = payload.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("completion_tokens_details")
    details = details if isinstance(details, dict) else {}
    fields = {
        "input_tokens": (usage.get("prompt_tokens"), "native_tokens_prompt", True),
        "output_tokens": (usage.get("completion_tokens"), "native_tokens_completion", True),
        "reasoning_tokens": (details.get("reasoning_tokens"), "native_tokens_reasoning", True),
        "cost": (usage.get("cost"), "total_cost", False),
    }
    usage_names = {
        "input_tokens": "usage.prompt_tokens", "output_tokens": "usage.completion_tokens",
        "reasoning_tokens": "usage.completion_tokens_details.reasoning_tokens", "cost": "usage.cost",
    }
    metrics, sources = {}, {}
    for name, (value, _, integer) in fields.items():
        metrics[name] = nonnegative_number(value, integer=integer)
        sources[name] = usage_names[name] if metrics[name] is not None else None
    required = ("input_tokens", "output_tokens", "cost")
    metadata, queries = {}, []
    if any(metrics[name] is None for name in required) and isinstance(payload.get("id"), str):
        metadata, queries = lookup_generation(client, payload["id"])
        for name, (_, native, integer) in fields.items():
            if metrics[name] is None:
                metrics[name] = nonnegative_number(metadata.get(native), integer=integer)
                if metrics[name] is not None:
                    sources[name] = f"generation.{native}"
    missing = [name for name in required if metrics[name] is None]
    return metrics, sources, metadata, queries, missing


def read_completion(payload):
    """Return (text, finish_reason, problem) without interpreting the answer."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None, None, "API returned no completion choices"
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    message = choice.get("message")
    if not isinstance(message, dict):
        return None, finish_reason, "API returned no assistant message"
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part["text"] for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    if not (isinstance(content, str) and content) and message.get("refusal"):
        content = str(message["refusal"])
    if content is not None and not isinstance(content, str):
        return None, finish_reason, "API returned non-text content"
    if choice.get("error"):
        return content, finish_reason, f"choice error: {choice['error']}"
    return content, finish_reason, None


def retry_delay(attempt, response=None):
    if response is not None:
        headers = {}
        try:
            payload = response.json()
            error = payload.get("error") if isinstance(payload, dict) else None
            metadata = error.get("metadata") if isinstance(error, dict) else None
            embedded = metadata.get("headers") if isinstance(metadata, dict) else None
            if isinstance(embedded, dict):
                headers.update({key.lower(): str(value) for key, value in embedded.items()})
        except ValueError:
            pass
        headers.update(response.headers)
        header = headers.get("retry-after")
        if header:
            try:
                return min(MAX_RETRY_DELAY, max(0.0, float(header)))
            except ValueError:
                try:
                    return min(MAX_RETRY_DELAY, max(0.0, parsedate_to_datetime(header).timestamp() - time.time()))
                except (TypeError, ValueError):
                    pass
        reset = nonnegative_number(headers.get("x-ratelimit-reset"))
        if reset is not None:
            reset = reset / 1000 if reset > 100_000_000_000 else reset
            remaining = reset - time.time()
            if remaining > 0:
                return min(MAX_RETRY_DELAY, remaining + 1)
    return min(60.0, 2.0 * 2 ** (attempt - 1)) * random.uniform(0.5, 1.0)


def produced_no_answer(response):
    """True for responses that carry no model output and may be retried safely."""
    if response.status_code in RETRYABLE_STATUS:
        return True
    if response.status_code != 200:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    if not isinstance(payload, dict) or payload.get("choices") or not isinstance(payload.get("error"), dict):
        return False
    return payload["error"].get("code") in RETRYABLE_STATUS


def post_with_retries(client, request, max_attempts, stop=None, pacer=None):
    """POST until a model response arrives; retry only failures that carry no answer."""
    log = []
    response = exception = None
    for attempt in range(1, max_attempts + 1):
        if stop is not None and stop.is_set():
            break
        if pacer is not None and not pacer.wait(stop):
            break
        response = exception = None
        started = time.perf_counter()
        try:
            response = client.post("chat/completions", json=request)
        except httpx.RequestError as exc:
            exception = exc
            log.append({"attempt": attempt, "error": f"{type(exc).__name__}: {exc}",
                        "latency": round(time.perf_counter() - started, 6)})
            retry = attempt < max_attempts
        else:
            log.append({"attempt": attempt, "status": response.status_code,
                        "latency": round(time.perf_counter() - started, 6)})
            retry = attempt < max_attempts and produced_no_answer(response)
        if not retry or (stop is not None and stop.is_set()):
            break
        delay = retry_delay(attempt, response)
        if pacer is not None and response is not None and response.status_code == 429:
            pacer.defer(delay)
        if stop is None:
            time.sleep(delay)
        elif stop.wait(delay):
            break
    return response, exception, log


def collect_one(client, question, model, messages, config, stop=None, pacer=None):
    row = dict.fromkeys(FIELDS)
    row.update({
        "question_id": question["question_id"], "subject": question["subject"],
        "question_type": question["question_type"], "model": model["id"],
        "correct_answer": question["correct_answer"],
    })
    parameters = request_parameters(config, model)
    request = {
        **parameters, "model": model["id"], "messages": messages,
        "max_tokens": config["max_output_tokens"], "stream": False,
    }
    response, exception, transport_log = post_with_retries(client, request, config["max_attempts"], stop, pacer)
    if not transport_log:
        return None  # Cancelled before any request was sent; leave the pair pending.
    row["latency"] = transport_log[-1]["latency"]
    payload, status, errors = None, None, []
    if response is None:
        errors.append(f"api_error: {type(exception).__name__}: {exception} (after {len(transport_log)} attempts)")
    else:
        status = response.status_code
        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        if not isinstance(payload, dict):
            row["raw_response"] = response.text
            errors.append(f"api_error: HTTP {status}; response is not a JSON object")
        elif status >= 400 or (payload.get("error") and not payload.get("choices")):
            errors.append(f"api_error: HTTP {status}; {payload.get('error', payload)} (after {len(transport_log)} attempts)")
        else:
            row["raw_response"], row["finish_reason"], problem = read_completion(payload)
            if problem:
                errors.append(f"api_error: {problem}")

    sources, generation, queries = {}, {}, []
    if isinstance(payload, dict) and status is not None and status < 400:
        metrics, sources, generation, queries, missing = reported_metrics(client, payload)
        row.update(metrics)
        if missing and not errors:
            errors.append("accounting_error: API did not report " + ", ".join(missing))
    row["error"] = "; ".join(errors) if errors else None
    return {
        "kind": "completion", "recorded_at": utc_now(), "result": row, "http_status": status,
        "request_parameters": parameters, "transport_attempts": transport_log,
        "response": payload, "metric_sources": sources,
        "generation_metadata": generation, "accounting_queries": queries,
    }


def refresh_accounting(client, journal_path, records):
    """Re-query billing (never the model) for rows whose only problem is missing accounting."""
    refreshed = 0
    for record in records.values():
        row = record["result"]
        response = record.get("response")
        if not (row["error"] or "").startswith("accounting_error") or not isinstance(response, dict):
            continue
        if not isinstance(response.get("id"), str):
            continue
        metrics, sources, generation, queries, missing = reported_metrics(client, response)
        updated = {**row, **metrics}
        updated["error"] = "accounting_error: API did not report " + ", ".join(missing) if missing else None
        append_attempt(journal_path, {
            "kind": "accounting_refresh", "recorded_at": utc_now(), "generation_id": response["id"],
            "result": updated, "metric_sources": sources, "generation_metadata": generation,
            "accounting_queries": queries,
        })
        refreshed += int(not missing)
    return refreshed


def format_duration(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def collect_dataset(
    data_dir, output_dir, config_path, model_ids=None, limit=None,
    dry_run=False, retry_errors=False, workers_per_model=None, requests_per_minute=None,
):
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    config = load_config(config_path)
    models = select_models(config, model_ids)
    workers = workers_per_model or config["concurrency_per_model"]
    if workers <= 0:
        raise ValueError("workers_per_model must be positive")
    if requests_per_minute is not None and (type(requests_per_minute) is not int or requests_per_minute <= 0):
        raise ValueError("requests_per_minute must be a positive integer")
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
        "request_parameters": config["request_parameters"], "models": config["models"],
        "prompt_version": PROMPT_VERSION,
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
            records = latest_records(journal_path)
            if any((record["result"]["error"] or "").startswith("accounting_error") for record in records.values()):
                print(f"Refreshed billing for {refresh_accounting(client, journal_path, records)} recorded responses")
            pending = [
                (question, model) for question in questions for model in models
                if (question["question_id"], model["id"]) not in records
                or (retry_errors and is_api_error(records[(question["question_id"], model["id"])]["result"])
                    and not records[(question["question_id"], model["id"])]["result"]["raw_response"])
            ]
            skipped = len(questions) * len(models) - len(pending)
            print(f"Pending {len(pending)} pairs; skipping {skipped} previously recorded pairs", flush=True)
            if not pending:
                if journal_path.exists():
                    export_results(output_dir)
                return {"requested": 0, "skipped": skipped, "errors": 0, "not_sent": 0}
            snapshot = fetch_catalog(client, models, config)
            print_catalog(snapshot)
            snapshot_name = "catalog_" + utc_now().replace(":", "-") + ".json"
            write_json(output_dir / "catalogs" / snapshot_name, snapshot)
            print(f"Sending with {workers} concurrent requests per model ({workers * len(models)} total)", flush=True)
            pacers = {model["id"]: RequestPacer(requests_per_minute) for model in models} if requests_per_minute else {}
            if requests_per_minute:
                print(f"Pacing all POST attempts at no more than {requests_per_minute} per minute per model", flush=True)

            stop = threading.Event()
            counters = {"requested": 0, "errors": 0, "cost": 0.0, "announced": False}
            started = time.monotonic()

            def task(question, model):
                if stop.is_set():
                    return None
                attempt = collect_one(client, question, model, build_messages(question, data_dir), config, stop,
                                      pacers.get(model["id"]))
                if attempt is not None and attempt["http_status"] in STOP_STATUS:
                    # Set in the worker so no other worker starts a new request first.
                    stop.set()
                return attempt

            def record(attempt):
                attempt["catalog_snapshot"] = snapshot_name
                append_attempt(journal_path, attempt)
                row = attempt["result"]
                counters["requested"] += 1
                counters["errors"] += int(row["error"] is not None)
                counters["cost"] += row["cost"] or 0
                done = counters["requested"]
                elapsed = time.monotonic() - started
                eta = elapsed / done * (len(pending) - done)
                state = f"error: {row['error'][:80]}" if row["error"] else f"finish={row['finish_reason']}"
                print(
                    f"[{done}/{len(pending)}] {row['question_id']} | {row['model']} | {state} | "
                    f"cost={row['cost']} | elapsed {format_duration(elapsed)} ETA {format_duration(eta)} | "
                    f"known cost ${counters['cost']:.4f}",
                    flush=True,
                )
                if attempt["http_status"] in STOP_STATUS and not counters["announced"]:
                    counters["announced"] = True
                    stop.set()
                    print(f"Stopping on authentication/credit/access error: {row['error']}", flush=True)

            executors = {
                model["id"]: ThreadPoolExecutor(max_workers=workers, thread_name_prefix=model["id"].replace("/", "_"))
                for model in models
            }
            outstanding = {executors[model["id"]].submit(task, question, model) for question, model in pending}
            try:
                while outstanding:
                    finished, _ = wait(outstanding, return_when=FIRST_COMPLETED)
                    for future in finished:
                        outstanding.discard(future)
                        attempt = future.result()
                        if attempt is not None:
                            record(attempt)
            except BaseException:
                # Requests already sent may be billed: record them before re-raising.
                stop.set()
                running = [future for future in outstanding if not future.cancel()]
                if running:
                    print(f"\nWaiting for {len(running)} in-flight requests so they are recorded "
                          "(press Ctrl+C again to abandon them)", flush=True)
                    for future in running:
                        try:
                            attempt = future.result()
                        except Exception:
                            continue
                        if attempt is not None:
                            record(attempt)
                raise
            finally:
                stop.set()
                for executor in executors.values():
                    executor.shutdown(wait=False, cancel_futures=True)
                # Derive exports from the durable journal, also after Ctrl+C.
                if journal_path.exists():
                    export_results(output_dir)
    not_sent = len(pending) - counters["requested"]
    if not_sent:
        print(f"{not_sent} pairs were not sent; run the same command again to continue")
    return {"requested": counters["requested"], "skipped": skipped, "errors": counters["errors"], "not_sent": not_sent}
