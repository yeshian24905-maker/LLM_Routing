# LLM_Routing

FT-UCB routes all **3000 questions with four saved model outcomes each** in the supplied `result.csv`. It selects one model per question and uses only that model's saved outcome as feedback. No model APIs are called.

See the [bandit guide](bandit/README.md) for FT-UCB (paper Algorithm 6) at four feasibility thresholds, per-round routing logs, cost/quality regret, reports and plots. No model APIs are called.

Run one 3000-round offline replay per threshold (89%, 85%, 80%, 82.5%):

```powershell
python -m bandit --results-csv result.csv --scores-csv bandit/output/ft-ucb/scored_results.csv --expected-prompts 3000 --empirical-costs --output-dir bandit/output/ft-ucb-verified
```

The optional archive preserves verified saved labels for its 1000 questions; it does not limit the run to that subset. The remaining 2000 questions are scored against the raw CSV reference answers, with review flags for open responses and unresolved choice text. `mu0` controls the per-round UCB candidate set. The original MMMU folder mode is still available via `--data-dir`.

The latest local 3000-question [report](bandit/output/ft-ucb-verified/report.md), [regret plot](bandit/output/ft-ucb-verified/regret.png), and [arm selection plot](bandit/output/ft-ucb-verified/arm_selections.png) are written to the separate verification directory. The archived 1000-question scores remain unchanged in `bandit/output/ft-ucb/`.
