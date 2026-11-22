"""Report and publication-exportable plots for the FT-UCB experiment."""

PAPER = "https://users.ece.cmu.edu/~oyagan/Conferences/ICLR2025.pdf"


def number(value, digits=6):
    return "N/A" if value is None else f"{value:.{digits}f}"


def make_ft_report(summary, runs):
    models = summary["models"]
    primary = [run for run in runs if run["seed"] == summary["primary_seed"]]
    lines = [
        "# FT-UCB offline routing experiment", "",
        (f"Questions: `{summary['questions_jsonl']}`; SHA-256: `{summary['questions_sha256']}`."
         if summary.get("questions_jsonl") else
         "Question set: all distinct question IDs in the supplied raw results CSV. "
         "An optional score archive supplies labels only; it never reduces the question set."), "",
        f"Saved model outcomes: `{summary['input_csv']}`; SHA-256: `{summary['input_sha256']}`.", "",
        f"{summary['prompts']} prompts x {len(models)} arms; {summary['repeats']} shuffled replays. "
        f"The primary run uses seed {summary['primary_seed']}. No new model calls are made.", "",
        f"Each threshold has {summary['prompts']} decision rounds, one per MMMU question. "
        "Saved outcomes for four arms are a lookup/evaluation table, not extra decision rounds.", "",
        f"[Algorithm 6, Appendix H]({PAPER}#page=54): initialize each arm once in ascending "
        "fixed-cost order; then choose the cheapest arm whose empirical mean plus "
        "sqrt(2 ln(t) / n) reaches mu0. If no arm qualifies, sample uniformly. "
        "The bound is not clipped to 1. Cost ties use model name order.", "",
        "Only the selected arm's outcome updates the policy. Each prompt occurs once per replay "
        "with immediate labeled feedback; thresholds share prompt order for each seed.", "",
        "mu0 is an input to the UCB candidate-set decision at every round, not an answer label or "
        "a final accuracy cutoff. Answers are compared with the supplied dataset's reference answers. "
        "Any final accuracy >= mu0 field is a descriptive diagnostic only; it never controls routing.", "",
        f"Fixed costs used by the policy: **{summary['cost_source']}**.", "",
    ]
    if summary.get("scores_csv"):
        lines.extend([
            f"Verified archived labels: `{summary['scores_csv']}`; SHA-256: `{summary['scores_sha256']}`. "
            f"Reused {summary['grading']['archived_rows_reused']} labels after matching every original raw field "
            "(only line endings may differ). Original review flags are retained. "
            f"The other {summary['grading']['raw_csv_rows_scored']} responses are scored from the supplied CSV. "
            "Numerical/text-only choices cannot be resolved when the raw CSV lacks option text.", "",
        ])
    if summary["cost_source"] != "given fixed costs":
        lines.extend([
            "Costs are calibrated from all known CSV costs, excluding missing entries, and held fixed. "
            "This uses future cost metadata and approximates the paper's known-cost assumption. "
            "Use --arm-costs for independently specified costs.", "",
        ])
    else:
        lines.extend([
            f"Cost file: `{summary['arm_costs_json']}`; "
            f"SHA-256: `{summary['arm_costs_sha256']}`.", "",
        ])
    lines.extend([
        "## Arm statistics (full-data evaluation)", "",
        "| Model | Accuracy | Fixed cost, USD/call | Mean known CSV cost | Missing CSV costs |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for model in models:
        stats = summary["arm_statistics"][model]
        lines.append(
            f"| {model} | {stats['accuracy']:.2%} | {summary['arm_costs_usd'][model]:.9f} "
            f"| {number(stats['mean_known_cost_usd'], 9)} | {stats['unknown_cost_count']} |"
        )
    lines.extend([
        "", "## Primary run: totals", "",
        "| mu0 | Correct | Accuracy | Total CSV cost, USD | Known cost, USD | Missing costs "
        "| Fixed-cost total, USD | Cost regret, USD | Quality regret | Cheapest feasible model |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for run in primary:
        lines.append(
            f"| {run['threshold']:.1%} | {run['correct']} | {run['accuracy']:.2%} "
            f"| {number(run['total_cost_usd'])} | {run['known_cost_usd']:.6f} "
            f"| {run['unknown_cost_count']} | {run['fixed_cost_total_usd']:.6f} "
            f"| {number(run['cost_regret_usd'])} | {run['quality_regret']:.6f} "
            f"| {run['optimal_feasible_model'] or 'INFEASIBLE'} |"
        )
    lines.extend([
        "", "## Primary run: arm choices", "",
        "| mu0 | " + " | ".join(models) + " | Total |",
        "| --- | " + " | ".join("---:" for _ in models) + " | ---: |",
    ])
    for run in primary:
        counts = run["model_selections"]
        lines.append(
            f"| {run['threshold']:.1%} | " + " | ".join(str(counts[arm]) for arm in models)
            + f" | {sum(counts.values())} |"
        )
    lines.extend(["", "![Primary-run arm selection counts](arm_selections.png)", ""])
    if summary["repeats"] > 1:
        lines.extend([
            "## Repeated replays", "",
            "| mu0 | Accuracy (mean +/- SD) | Known CSV cost (mean +/- SD), USD "
            "| Cost regret (mean +/- SD), USD | Quality regret (mean +/- SD) | Runs meeting mu0 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for scenario in summary["scenarios"]:
            values = scenario["metrics"]
            accuracy = values["accuracy"]
            cells = [f"{accuracy['mean']:.2%} +/- {accuracy['std']:.2%}"]
            for field in ("known_cost_usd", "cost_regret_usd", "quality_regret"):
                metric = values[field]
                cells.append(f"{number(metric['mean'])} +/- {number(metric['std'])}")
            lines.append(
                f"| {scenario['threshold']:.1%} | " + " | ".join(cells)
                + f" | {scenario['runs_meeting_threshold']}/{summary['repeats']} |"
            )
    lines.extend([
        "", "## Regret definitions", "",
        f"Following [Section 1.1]({PAPER}#page=2), let mu_i be the full-data accuracy "
        "estimate, c_i the fixed policy cost, and a* the cheapest arm with mu_i >= mu0. "
        "The plotted empirical path regrets are:", "",
        "```text",
        "R_cost(t)    = sum_{s=1}^t max(0, c[chosen_s] - c[a*])",
        "R_quality(t) = sum_{s=1}^t max(0, mu0 - mu[chosen_s])",
        "```", "",
        "Cost regret is in USD; quality regret is in expected-correct-answer equivalents. "
        "These plug-in gaps use full-data arm means for evaluation only, never policy updates. "
        "An infeasible threshold has cost regret N/A; quality regret remains defined.", "",
        "![Cumulative empirical cost and quality regrets](regret.png)", "",
        "## Discussion and limitations", "",
        f"FT-UCB need not reach mu0 in {summary['prompts']} rounds: optimism can keep cheap infeasible arms "
        "in the candidate set. Small cost regret alone does not establish quality feasibility.", "",
        "Actual costs sum selected CSV entries. With missing costs, the total is N/A and the "
        "known sum is a lower bound. Fixed-cost totals/regret use c_i, not per-prompt token costs.", "",
        f"Scorer: {summary['scorer']}. {summary['grading']['needs_review']} observations "
        "are flagged for manual review in scored_results.csv. Automatic scores and all "
        "quality comparisons remain provisional until reviewed labels are supplied.", "",
        "Curves show one sampled path at each threshold by default. With multiple replays, "
        "they show means +/- sample SD, approximating path expectations. SD measures "
        "order/randomness sensitivity, not uncertainty on unseen prompts.", "",
        "Full per-run counts and totals are in runs.csv; each decision is in trace.csv. "
        "Plots are exported as PNG/PDF for slides.", "",
    ])
    return "\n".join(lines)


def plot_ft(output_dir, summary, runs, curves):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for scenario in summary["scenarios"]:
        rows = [row for row in curves if row["threshold"] == scenario["threshold"]]
        t = [row["step"] for row in rows]
        label = f"mu0 = {scenario['threshold']:.1%}"
        for axis, prefix in zip(axes, ("cost_regret", "quality_regret")):
            suffix = "_usd" if prefix == "cost_regret" else ""
            means = [row[f"{prefix}_mean{suffix}"] for row in rows]
            sds = [row[f"{prefix}_std{suffix}"] for row in rows]
            if means[0] is None:
                axis.plot([], [], linestyle="none", label=f"{label} (infeasible; N/A)")
                continue
            line, = axis.plot(t, means, label=label, linewidth=1.8)
            if summary["repeats"] > 1:
                axis.fill_between(
                    t, [max(0.0, m - sd) for m, sd in zip(means, sds)],
                    [m + sd for m, sd in zip(means, sds)], color=line.get_color(), alpha=0.13,
                )
    for axis, title, unit in zip(
        axes, ("Empirical cost regret", "Empirical quality regret"),
        ("Cumulative regret (USD)", "Cumulative expected-correct deficit"),
    ):
        axis.set(title=title, xlabel="Decision round t", ylabel=unit)
        axis.set_xlim(1, max(2, summary["prompts"]))
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    title = f"mean +/- SD over {summary['repeats']} replays" if summary["repeats"] > 1 else "one run per threshold"
    fig.suptitle(f"FT-UCB: {title}")
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"regret.{extension}", dpi=180)
    plt.close(fig)

    primary = [run for run in runs if run["seed"] == summary["primary_seed"]]
    fig, axis = plt.subplots(figsize=(9, 5.3), layout="constrained")
    positions, bottom = list(range(len(primary))), [0] * len(primary)
    for arm in summary["models"]:
        counts = [run["model_selections"][arm] for run in primary]
        bars = axis.bar(positions, counts, bottom=bottom, label=arm)
        for bar, count, offset in zip(bars, counts, bottom):
            if count >= summary["prompts"] * 0.04:
                axis.text(bar.get_x() + bar.get_width() / 2, offset + count / 2,
                          str(count), ha="center", va="center", fontsize=9)
        bottom = [old + count for old, count in zip(bottom, counts)]
    axis.set_xticks(positions, [f"{run['threshold']:.1%}" for run in primary])
    axis.set(xlabel="Feasibility threshold mu0", ylabel="Number of arm choices",
             title=f"FT-UCB: {summary['prompts']} decisions, primary seed {summary['primary_seed']}")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8, ncol=2)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"arm_selections.{extension}", dpi=180)
    plt.close(fig)
