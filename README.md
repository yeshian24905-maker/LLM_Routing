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

The code in `bandit/` needs matplotlib and scipy. The `dev` group adds numpy and pandas for experiments. `uv sync` installs the `dev` group, and `uv sync --no-dev` does not. The `features` group holds PyTorch and the other packages of `question_features` (see [Learned question features](#learned-question-features)). Install it with `uv sync --group features`. A plain `uv sync` removes it again, but `uv run` keeps it. To add a package, run `uv add <package>`, or `uv add --dev <package>` for a package that only experiments use. Commit `pyproject.toml` and `uv.lock` together.

Run the unit tests:

```powershell
python -m unittest discover -s bandit/tests
python -m unittest discover -s question_features/tests
```

The second command needs the `features` group.

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

## Learned question features

The `question_features` package predicts, for each question, the probability that each model answers it correctly. It reads the question text, the images and simple counts such as the number of images and formulas. The predictions give `bandit.domain_compare` two new inputs: an accuracy prior for each question, and learned groups of questions.

The extra packages include PyTorch, so they are in their own `features` group. Install them:

```powershell
uv sync --group features
```

`uv run` keeps the group. A plain `uv sync` removes it, so after a plain `uv sync`, run `uv sync --group features` again.

Run these four steps in order:

1. `python -m question_features fetch` downloads the 3000 MMMU test questions of `results_3000.csv` from the public `MMMU/MMMU` dataset on Hugging Face. It needs network access but no login. `REVISION` in `question_features/mmmu.py` fixes the dataset version. The parquet files, about 3.3 GB, stay in the Hugging Face cache. The questions and images, about 1 GB, go to `data/mmmu/`. Git ignores `data/`, so each user downloads the data again.
2. `python -m question_features embed` writes text embeddings (`BAAI/bge-base-en-v1.5`) and image embeddings (CLIP ViT-B/32) to `data/mmmu/embeddings.npz`. It uses the Apple GPU (MPS) when one is available, and the CPU otherwise. On an M4 it takes less than 2 minutes.
3. `python -m question_features predict` fits five predictors of the accuracy of each model:
   - `pooled`: the mean accuracy of the model.
   - `subject`: the mean accuracy of the model in the subject, pulled toward `pooled` with a weight of 20 questions.
   - `handcrafted`: a logistic regression on the subject and the question counts.
   - `text`: `handcrafted` plus a 32-dimension projection of the text embedding.
   - `text_image`: `text` plus a 32-dimension projection of the image embedding.

   The projections use no labels. The step uses 5-fold cross-fitting, so the prediction for a question comes from a fit that did not see the outcomes of that question. It writes `data/mmmu/priors.csv` (one row per question and predictor, one column per model) and `data/mmmu/predictor_metrics.json`.
4. `python -m question_features cluster` groups the questions with k-means on their four `text_image` predicted accuracies, for k = 8, 16 and 30. For each k it writes `groups_text_image_k<k>.csv` and `cluster_summary_text_image_k<k>.csv` to `data/mmmu/`.

Then use the predictions in the comparison:

```powershell
python -m bandit.domain_compare --empirical-costs --priors data/mmmu/priors.csv
python -m bandit.domain_compare --empirical-costs --groups data/mmmu/groups_text_image_k16.csv
```

- `--priors` adds the `prior_<predictor>_k<kappa>` policies, for example `prior_text_image_kinf`. Each choice starts from the predicted accuracy of the question, and λ changes as in the domain policies. `--prior-predictors` selects the predictors (default `subject text_image`).
- `--prior-kappas` (default `20 inf`) sets how strongly a policy trusts the predictions, as a number of outcomes. The policy corrects the predictions of each model in each domain with the outcomes it sees there. After kappa outcomes, the correction has half the weight. With `inf`, the policy uses the predictions without a correction.
- `--groups` uses the learned group instead of the subject as the domain of each question, for every policy and for the optimum. With `--priors`, the correction is then per learned group.

Each predictor learns from the saved outcomes of all four models on the other 80% of the questions. A router in service needs such labelled data before it starts.

## Acknowledgement

The development of this project is assisted by LLMs and AI agents.
