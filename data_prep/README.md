# MMMU Data Preparation

Run `python -m data_prep` from the project root. The pipeline downloads MMMU questions and images, queries four models, scores their answers, and exports one observation per question/model pair.

The implementation uses five Python modules:

| File | Responsibility |
| --- | --- |
| `__main__.py` | Command-line entry point for all four commands |
| `dataset.py` | Downloads, validation, image handling, and multimodal prompts |
| `collector.py` | Model requests, actual billing lookups, and resumable collection |
| `scoring.py` | Answer extraction and scoring |
| `common.py` | Configuration, file I/O, journal recovery, and result exports |

Model IDs and request settings are in `config.json`. Prices are read from the live model catalog; recorded costs come from actual API billing.

## Usage

Python 3.10 or later is required. Start with a small sample:

```powershell
python -m pip install -r data_prep/requirements.txt
python -m data_prep check-models
python -m data_prep download --splits dev --subjects Accounting Physics --limit-per-subject 1 --output-dir data_prep/output/smoke
python -m data_prep run --data-dir data_prep/output/smoke --dry-run

# Real collection: 2 questions x 4 models = 8 completion requests
$env:OPENROUTER_API_KEY = "your-api-key"
python -m data_prep run --data-dir data_prep/output/smoke --delay 1
```

`--dry-run` runs offline, makes no model requests, and creates no fabricated results. The API key is read from the environment and is never written to artifacts. The scripts do not load `.env` automatically.

Prepare the validation split for all 30 subjects:

```powershell
python -m data_prep download
python -m data_prep run --dry-run
python -m data_prep run --delay 1
```

The default data directory is `data_prep/output/mmmu_validation`. Results are saved under its `runs/default/` subdirectory.

Use `python -m data_prep <command> --help` for all options. Common examples:

- `download --splits dev validation test --output-dir ...` combines splits.
- `download --subjects Accounting Physics Art_Theory` selects subjects; names are case-sensitive.
- `download --limit-per-subject N` takes the first N questions per subject/split in source order.
- `download --revision ...` selects a version, which is resolved to a fixed commit SHA.
- `run --models anthropic/claude-haiku-5.5 --limit N` selects a model and the first N prepared questions.
- `run --output-dir ...` keeps a separate experiment in its own directory.

## Output

`results.csv` and `results.jsonl` contain the same 12 fields:

| Field | Meaning |
| --- | --- |
| `question_id` | Original MMMU identifier |
| `subject` | Subject name |
| `model` | Full model ID used in the request |
| `prediction` | Parsed option letter or final open-ended answer |
| `correct_answer` | Ground-truth answer |
| `correct` | Integer 0 or 1; API or answer-parsing failures receive 0 |
| `input_tokens` | Reported input usage, including images |
| `output_tokens` | Reported output usage; reasoning tokens are not added again |
| `cost` | Actual reported completion cost in USD |
| `latency` | Non-streaming completion HTTP request duration in seconds |
| `raw_response` | Unmodified model response text |
| `error` | API, parsing, or missing-accounting error; null on success |

Cost is taken from `usage.cost`, with `/generation`'s `total_cost` as a fallback. Token counts use `usage.prompt_tokens/completion_tokens`, falling back to `native_tokens_prompt/native_tokens_completion`. Missing values remain `null` and are recorded as errors. Actual zero cost remains `0`. CSV represents null as an empty cell and answer lists as JSON strings.

Latency excludes image encoding, delays between requests, and subsequent billing lookups. Missing accounting data does not change a completed answer's grade, so `correct=1` can coexist with an `accounting_error`.

Multiple-choice answers must identify a unique option; the parser never guesses. Open-ended scoring uses only the final answer, with case-insensitive text matching at word boundaries and numeric values rounded to two decimal places. It supports units, thousands separators, scientific notation, and answer lists. These deterministic parsing rules differ from the official evaluator's free-text fallback behavior.

`questions.jsonl` stores question text, options, labels, image paths, and checksums. Images are inserted at their `<image n>` markers and attached once each; other nonempty images are also included. Ground-truth answers and explanations are excluded from model prompts. Full API responses, billing lookups, and field sources are stored in `attempts.jsonl`; model catalog snapshots are stored under `catalogs/`.

## Resume, Retry, and Export

Running the same `run` command skips recorded pairs, including error rows. Each completed request is flushed to the journal. CSV/JSONL retain the latest result for each `(question_id, model)` pair. A file lock prevents concurrent collection into the same directory, and data/configuration fingerprints prevent mixing experiments. Completed datasets are never overwritten.

Changes to scoring rules, prompts, model settings, or prepared questions require a new `--output-dir`. The current scoring version is `deterministic-mmmu-v2`; existing journals retain their original grades.

```powershell
# Re-query error rows explicitly; these requests may incur new charges
python -m data_prep run --data-dir data_prep/output/smoke --retry-errors --delay 1

# Rebuild exports from the journal without querying models
python -m data_prep export --run-dir data_prep/output/smoke/runs/default

# Offline tests, including a simulated download -> four-model collection -> export workflow
python -m unittest discover -s data_prep/tests -v
```

Older attempts remain in the journal. `summary.json` reports the latest observations' costs, cumulative known costs across all attempts, and the number of attempts with unknown costs. `--retry-errors` also re-queries rows whose only error is missing accounting data.

HTTP 401/402/403 responses stop collection after saving the current error. Other individual errors are recorded and collection continues. A collection invocation with errors returns exit code 1. If the process exits after the server completes a request but before the local journal is written, resuming cannot determine whether that request was billed. Filter error rows before training so API failures are not treated as model capability failures.

References: [MMMU dataset](https://huggingface.co/datasets/MMMU/MMMU), [official evaluator](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/eval_utils.py), and [OpenRouter usage and billing](https://openrouter.ai/docs/cookbook/administration/usage-accounting).
