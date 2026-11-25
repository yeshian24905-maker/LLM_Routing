# LLM_Routing

FT-UCB routes all **3000 questions with four saved model outcomes each** in the supplied `results_3000.csv`. It selects one model per question and uses only that model's saved outcome as feedback. No model APIs are called.

See the [bandit guide](bandit/README.md) for FT-UCB (paper Algorithm 6) at four feasibility thresholds, per-round routing logs, cost/quality regret, reports and plots. No model APIs are called.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) to manage Python and the packages. `pyproject.toml` lists the dependencies, and `uv.lock` pins their versions. The code needs Python 3.10 or later. `.python-version` pins Python 3.15, so that every computer uses the same interpreter.

Run all commands in the project root, which is the directory that contains this README and `pyproject.toml`. This also applies to the commands in the bandit guide.

1. Install uv. Its [installation guide](https://docs.astral.sh/uv/getting-started/installation/) gives the command for each operating system.
2. Run `uv sync`. This command creates `.venv` with Python 3.15 and all packages, or updates an existing `.venv`. If Python 3.15 is not on the computer, uv downloads it.
3. Check the environment: `uv run python -m bandit --help` prints the usage of the command-line tool.

The usage commands in this guide and in the [bandit guide](bandit/README.md) start with `python`. Run them in one of these two ways:

- Put `uv run` before the command, for example `uv run python -m bandit --help`.
- Activate the environment first, then type the command as shown. To activate, run `source .venv/bin/activate` on macOS and Linux, or `.venv\Scripts\Activate.ps1` in Windows PowerShell.

Do not run the commands with the system `python3` of macOS. It is Python 3.9, which cannot import this code, and it does not have the packages.

The code in `bandit/` needs matplotlib and scipy. The `dev` group adds numpy and pandas for experiments. `uv sync` installs the `dev` group, and `uv sync --no-dev` does not. To add a package, run `uv add <package>`, or `uv add --dev <package>` for a package that only experiments use. Commit `pyproject.toml` and `uv.lock` together.

Run the unit tests:

```powershell
python -m unittest discover -s bandit/tests
```

## Usage

Run one 3000-round offline replay per threshold (89%, 85%, 80%, 82.5%):

```powershell
python -m bandit --results-csv results_3000.csv --scores-csv bandit/output/ft-ucb/scored_results.csv --expected-prompts 3000 --empirical-costs --output-dir bandit/output/ft-ucb-verified
```

The optional archive preserves verified saved labels for its 1000 questions; it does not limit the run to that subset. The remaining 2000 questions are scored against the raw CSV reference answers, with review flags for open responses and unresolved choice text. `mu0` controls the per-round UCB candidate set. The original MMMU folder mode is still available via `--data-dir`.

The latest local 3000-question [report](bandit/output/ft-ucb-verified/report.md), [regret plot](bandit/output/ft-ucb-verified/regret.png), and [arm selection plot](bandit/output/ft-ucb-verified/arm_selections.png) are written to the separate verification directory. The archived 1000-question scores remain unchanged in `bandit/output/ft-ucb/`.

Compare the original UCB with the three proposed confidence-bound changes:

```powershell
python -m bandit.compare --results-csv results_3000.csv --empirical-costs --repeats 30 --horizons 3000 10000
```

This tests tight UCB (`0.5 log(t)`), Bernoulli KL-UCB (`c=0`, with `c=3` as a sensitivity check), and LCB (the original radius with a minus sign). Each policy keeps the same cheapest-eligible-arm selection and uniform fallback. The 3000-round runs use every question once; 10000-round runs sample the same saved pool with replacement, with no new model calls. The command blocks network connections, writes paired results and primary-seed traces to `bandit/output/confidence-comparison/`, and exports [one English PDF](artifacts/Confidence_Methods_Comparison.pdf).

Compare routing by question domain with the FT baselines:

```powershell
python -m bandit.domain_compare --empirical-costs
```

The two domain policies use the subject of each question as its domain, and keep the overall accuracy at the threshold. For each question, they choose the model with the lowest `cost − λ × accuracy`. The accuracy of a model in a domain starts at the model's accuracy over all domains and moves to the domain's own value as data comes in. `λ` grows when the cumulative accuracy falls below the threshold. `pd_*` policies use an upper confidence bound, and `ts_*` policies use Thompson sampling. The command also reports the cheapest routing that reaches each threshold, with and without domains, from all saved outcomes. It writes the results and `tradeoff.png` to `bandit/output/domain-comparison/`. The [bandit guide](bandit/README.md) gives the details.

## Acknowledgement

The development of this project is assisted by LLMs and AI agents.
