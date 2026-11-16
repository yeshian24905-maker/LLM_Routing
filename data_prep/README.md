# MMMU Data Preparation

This package builds the routing dataset: **1000 MMMU questions × 4 models = 4000 raw observations**. Each model is asked each question **once**. The pipeline records the raw response next to the ground-truth answer. It does **not** parse, score, or re-ask; reviewing and grading happen afterwards, outside this package.

| File | Responsibility |
| --- | --- |
| `__main__.py` | Command-line entry point (`download`, `check-models`, `pilot`, `run`, `export`) |
| `dataset.py` | Seeded, subject-balanced sampling, image handling, multimodal prompts |
| `collector.py` | Parallel requests, retries for failed requests, billing lookups, resumable journal |
| `pilot.py` | Small paid trial on one model and a report with questions, options, images, and raw responses |
| `common.py` | Configuration, file I/O, journal recovery, exports and summaries |

## Setup

Python 3.10+ is required (developed with 3.15 via [uv](https://docs.astral.sh/uv/)). From the project root:

```bash
uv venv --python 3.15 .venv
uv pip install --python .venv -r data_prep/requirements.txt
source .venv/bin/activate
python -m unittest discover -s data_prep/tests      # offline, no network, no cost
```

The API key is read from the environment and never written to disk:

```bash
export OPENROUTER_API_KEY="..."
```

**Set a spending limit on this OpenRouter key in the OpenRouter dashboard before any paid step.** The code has no budget feature on purpose: when the key's limit is reached, OpenRouter answers HTTP 402 and the collector stops all workers.

On macOS, wrap long steps (download, full run) in `caffeinate -dims ...` so the machine does not sleep.

## Workflow

```bash
# 1. Download the sample (whole test split is ~3.3 GB because images are embedded; see "Dataset")
caffeinate -dims python -m data_prep download

# 2. Offline check of every prompt and image (no API calls)
python -m data_prep run --dry-run

# 3. Check model IDs, image support, request parameters, and live prices
python -m data_prep check-models

# 4. Pilot: cheapest model (by live price) on the first 10, then 100 questions
python -m data_prep pilot --n 10
python -m data_prep pilot --n 100
#    -> read data_prep/output/mmmu_test_sample/runs/pilot_<model>_<n>/pilot_report.md

# 5. Full collection: 1000 questions x 4 models
caffeinate -dims python -m data_prep run
```

Defaults: data in `data_prep/output/mmmu_test_sample/`, full run in its `runs/default/`. Use `python -m data_prep <command> --help` for all options, for example `run --models <id> --limit N`, `run --workers-per-model N`, `download --seed N --sample-size N|all --splits dev validation test`.

## Dataset

- **Source:** MMMU `test` split (10,500 questions, all with released answers), pinned to the resolved Hugging Face commit recorded in `dataset_manifest.json`.
- **Sampling:** 1000 questions split as evenly as possible over the 30 subjects (33 or 34 each). The seed (default `20261116`) decides both the questions and which subjects get the extra one, so the same command always gives the same sample. Unusable rows are skipped, listed in `skipped_rows`, and replaced by the next question in that subject's seeded order.
- **Order:** `questions.jsonl` is globally shuffled, so the first N questions (`--limit N`, `pilot --n N`) are a mixed-subject subset of the 1000 and the pilot's questions are part of the final sample.
- **Download size:** each subject's split is a single parquet row group, so rows cannot be fetched individually; the download reads whole files (~3.3 GB for `test`). Files are cached under the data directory and resumable.
- **Images:** original JPEG/PNG bytes are kept whenever possible. Images with real transparency are composited onto **white** (a plain RGB conversion turns transparent areas black). EXIF-rotated, CMYK and other formats are re-encoded; images over 7680 px per side or 3.75 MB are scaled down to stay within provider limits. Every change is recorded per image in `image_processing` and counted in `image_stats`.
- **Answers:** about 0.3% of `correct_answer` values are Python-list strings of accepted alternatives, e.g. `"['$Al_2S_3$', 'Al2S3']"`. They are stored unchanged; graders must treat them as "any of these".
- Ground-truth answers and explanations are never put into prompts.

## Collection rules

- **One question, one answer.** Whatever the model returns is recorded as is: empty answers, refusals, truncated output (`finish_reason=length`), answers without an `<answer>` tag. None of these are errors and none are re-asked.
- **Only requests with no model response are retried automatically**: network errors, timeouts, HTTP 408/429/5xx, and HTTP 200 bodies that only carry such an error. Up to `max_attempts` (5) tries with exponential backoff that honors `Retry-After`. Every try is logged in `transport_attempts`. A timed-out request may still have been processed and billed upstream; this cannot be detected.
- **HTTP 401/402/403** stops all workers; requests already in flight are still recorded.
- **Missing billing data** (`accounting_error`): only the billing endpoint is queried again (on the next `run`), never the model.
- `run --retry-errors` re-sends only pairs that ended with an `api_error` (no model response). Re-asking pairs that did get a response is a separate, later step after manual review.
- **Model settings** (`config.json`): all models share `reasoning.effort = "low"`, `max_tokens = 16384`, and a 600 s timeout. `concurrency_per_model` (8) requests run in parallel per model, 32 in total.

## Rate-Limited Accounts

Concurrency and requests per minute are different limits. Use `--requests-per-minute` to pace every POST attempt, including retries, across all workers for each selected model. On 429 errors, retries also honor `Retry-After` or `X-RateLimit-Reset`, including headers embedded in the error metadata.

For an account limited to 20 requests per minute, run at 15 per minute with one worker. To repair only API failures among the first 110 questions and refresh the pilot report:

```powershell
python -m data_prep pilot --model openai/gpt-5.6-luna --n 110 --output-dir data_prep/output/mmmu_test_sample/runs/sequential --retry-errors --requests-per-minute 15 --workers-per-model 1
```

Keep the original data directory and output directory. Existing responses are skipped; `--retry-errors` only selects API-error rows without captured response text. An absent `finish_reason` after HTTP 429 means no completion was received. To continue through all 1000 questions, use the same command with `--n 1000`. Pacing and worker options do not change the experiment identity, so the existing journal can be resumed.

## Output

`results.csv` / `results.jsonl` hold the latest observation per `(question_id, model)` with 13 fields:

| Field | Meaning |
| --- | --- |
| `question_id`, `subject`, `question_type` | From MMMU |
| `model` | Full model ID used in the request |
| `raw_response` | Model text exactly as returned (refusal text if the model refused; null if no response) |
| `correct_answer` | Ground-truth answer, unchanged |
| `finish_reason` | As reported, e.g. `stop`, `length` |
| `input_tokens`, `output_tokens` | Reported usage; output includes reasoning tokens |
| `reasoning_tokens` | Reported reasoning tokens (subset of output), null if not reported |
| `cost` | Actual billed USD (`usage.cost`, else `/generation` `total_cost`); 0 is a real value |
| `latency` | Seconds for the final HTTP attempt |
| `error` | `api_error: ...` (no usable response) or `accounting_error: ...`; null otherwise |

Other files in the run directory:

- `attempts.jsonl`: append-only journal holding the full API payloads (including reasoning text), request settings, retries, and billing queries.
- `summary.json`: per-model counts, `finish_reason` distribution, empty responses, tokens, cost, and latency percentiles. `all_requests` totals billing over every request, including re-sent ones.
- `run_manifest.json` and `catalogs/`: run settings and price snapshots.

The pilot's `pilot_report.md` includes the source question, labeled options, local images, ground-truth answer, and unmodified model response for each recorded pair. Open it in VS Code and press `Ctrl+Shift+V` to preview the images. Questions are joined from `questions.jsonl`; rebuilding the report uses existing records without model requests.

Re-running the same command resumes and skips recorded pairs. A file lock prevents two runs in one directory, and changing data, prompts, or `config.json` request settings requires a new `--output-dir`. `python -m data_prep export --run-dir DIR` rebuilds the exports from the journal without network access.

References: [MMMU dataset](https://huggingface.co/datasets/MMMU/MMMU), [OpenRouter usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting).
