# LLM_Routing

FT-UCB routes the 1000 prompts in `mmmu_test_sample/questions.jsonl`, reusing matching model outcomes from `mmmu_test_sample/runs/sequential/`. The dataset folder is local and excluded from Git.

See the [bandit guide](bandit/README.md) for FT-UCB (paper Algorithm 6) at four feasibility thresholds, per-round routing logs, cost/quality regret, reports and plots. No model APIs are called.

The latest experiment [report](bandit/output/ft-ucb/report.md), [regret plot](bandit/output/ft-ucb/regret.png), and [arm selection plot](bandit/output/ft-ucb/arm_selections.png) are versioned with the code. Re-running the experiment overwrites these results.
