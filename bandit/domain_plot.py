"""Cost-accuracy trade-off figure for the domain comparison."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SERIES = (
    ("ft_kl_ucb", "FT KL-UCB (global)", "#257de0", "o"),
    ("pd_k20_eta1_beta0.5", "Primal-dual UCB, κ=20", "#f47230", "s"),
    ("ts_k20_eta1", "Primal-dual Thompson, κ=20", "#00b894", "D"),
)
TEXT, MUTED, GRID = "#191919", "#666666", "#e6e6e6"


def plot_tradeoff(aggregates, path, thresholds=(0.825, 0.85, 0.89)):
    """One panel per horizon (rows) and threshold (columns): mean cost vs mean accuracy, ±1 SD."""
    horizons = sorted({row["horizon"] for row in aggregates})
    fig, axes = plt.subplots(len(horizons), len(thresholds), figsize=(4.2 * len(thresholds), 3.6 * len(horizons)),
                             squeeze=False, constrained_layout=True)
    for i, horizon in enumerate(horizons):
        for j, threshold in enumerate(thresholds):
            ax = axes[i][j]
            rows = {r["policy"]: r for r in aggregates if r["horizon"] == horizon and r["threshold"] == threshold}
            if not rows:
                ax.set_visible(False)
                continue
            ax.axhline(threshold, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
            ax.text(0.99, threshold, f"τ = {threshold:g}", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=8, color=MUTED)
            any_row = next(iter(rows.values()))
            for key, label, marker in (("oracle_global_cost_usd_per_1k", "Optimum, no domains", "o"),
                                       ("oracle_domain_cost_usd_per_1k", "Optimum, per domain", "*")):
                if any_row[key] is not None:
                    ax.scatter(any_row[key], threshold, s=90 if marker == "*" else 60, marker=marker,
                               facecolor=TEXT if marker == "*" else "none", edgecolor=TEXT,
                               linewidth=1.5, zorder=4, label=label)
            for policy, label, color, marker in SERIES:
                row = rows.get(policy)
                if row is None:
                    continue
                ax.errorbar(row["cost_usd_per_1k_mean"], row["accuracy_mean"],
                            xerr=row["cost_usd_per_1k_std"], yerr=row["accuracy_std"],
                            fmt=marker, markersize=8, color=color, markeredgecolor="white",
                            markeredgewidth=1.5, elinewidth=1.5, capsize=0, zorder=3, label=label)
            ax.set_title(f"{horizon:,} rounds, τ = {threshold:g}", fontsize=10, color=TEXT, loc="left")
            ax.grid(color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=8)
            if i == len(horizons) - 1:
                ax.set_xlabel("Cost (USD per 1000 questions)", fontsize=9, color=MUTED)
            if j == 0:
                ax.set_ylabel("Accuracy", fontsize=9, color=MUTED)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False, fontsize=9)
    fig.savefig(path, dpi=160)
    plt.close(fig)
