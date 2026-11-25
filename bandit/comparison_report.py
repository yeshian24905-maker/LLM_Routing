"""English PDF report for the paired confidence-bound experiments."""

import json
import math
from pathlib import Path

from .confidence import POLICIES


COLORS = {"ucb_original": "#607789", "ucb_tight": "#267C84", "kl_ucb": "#D37B29",
          "kl_ucb_c3": "#7764AF", "lcb": "#B54F61"}
PAPER = "https://users.ece.cmu.edu/~oyagan/Conferences/ICLR2025.pdf#page=54"
KL_PAPER = "https://proceedings.mlr.press/v19/garivier11a/garivier11a.pdf"


def render_comparison(summary, runs, aggregates, curves, pdf_path, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle
    from matplotlib.textpath import TextToPath

    pdf_path, output_dir = Path(pdf_path), Path(output_dir)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    width, height, left, right = 8.2677, 11.6929, .62, 7.65
    usable = right-left
    ink, muted, teal, pale, line = "#172B40", "#526778", "#247A81", "#F1F5F8", "#DCE4EB"
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.parse_math": False,
        "pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": line, "axes.labelcolor": muted,
        "xtick.color": muted, "ytick.color": muted})
    metric = TextToPath()
    horizons = summary["horizons"]
    page_count = len(horizons)+4
    policy_ids = list(summary["policies"])
    thresholds = summary["thresholds"]
    focus_threshold = max(thresholds)
    stats = {(r["horizon"], r["threshold"], r["policy"]): r for r in aggregates}

    def wrap(value, size, span, weight="normal"):
        fp = FontProperties(family="DejaVu Sans", size=size, weight=weight)
        result = []
        for block in str(value).split("\n"):
            row = ""
            for word in block.split():
                attempt = (row+" "+word).strip()
                if row and metric.get_text_width_height_descent(attempt, fp, False)[0] > span*72:
                    result.append(row)
                    row = word
                else:
                    row = attempt
            result.append(row)
        return result

    def text(fig, x, y, value, size=10, color=ink, weight="normal", ha="left", **kwargs):
        return fig.text(x/width, 1-y/height, str(value), fontsize=size, color=color,
            weight=weight, ha=ha, va="top", **kwargs)

    def para(fig, y, value, size=9.5, color=ink, x=left, span=usable):
        rows = wrap(value, size, span)
        for j, row in enumerate(rows):
            text(fig, x, y+j*size*1.43/72, row, size, color)
        return y+len(rows)*size*1.43/72

    def box(fig, x, y, span, tall, color=pale):
        fig.patches.append(Rectangle((x/width, 1-(y+tall)/height), span/width, tall/height,
            transform=fig.transFigure, facecolor=color, edgecolor="none", zorder=-1))

    def section(fig, y, value):
        text(fig, left, y, value, 12, ink, "bold")

    def base(number, title, subtitle):
        fig = plt.figure(figsize=(width, height), dpi=150)
        text(fig, left, .36, "CONFIDENCE-BOUND METHODS  /  OFFLINE COMPARISON", 8, teal, "bold")
        measured = metric.get_text_width_height_descent(title,
            FontProperties(family="DejaVu Sans", size=23, weight="bold"), False)[0]
        text(fig, left, .83, title, min(23, 23*usable*72/measured), ink, "bold")
        para(fig, 1.35, subtitle, 9.5, muted)
        fig.lines.append(plt.Line2D([left/width, right/width], [1-10.98/height]*2,
            transform=fig.transFigure, color=line, linewidth=.7))
        text(fig, left, 11.15,
            f"3,000 source questions  |  4 models  |  {summary['repeats']} paired seeds  |  No model API calls", 7, muted)
        text(fig, right, 11.15, f"{number:02d} / {page_count:02d}", 7, muted, ha="right")
        return fig

    def table(fig, y, headers, rows, widths, size=8, row_height=.32, highlight=None):
        hh = max(.50, max(len(wrap(h, 7.8, w-.13, "bold")) for h, w in zip(headers, widths))*.14+.17)
        box(fig, left, y, usable, hh, ink)
        x = left
        for i, (header, span) in enumerate(zip(headers, widths)):
            ha, px = ("left", x+.065) if i < 2 else ("right", x+span-.065)
            for j, row in enumerate(wrap(header, 7.8, span-.13, "bold")):
                text(fig, px, y+.08+j*.14, row, 7.8, "white", "bold", ha)
            x += span
        y += hh
        for i, row in enumerate(rows):
            blocks = [wrap(v, size, w-.13) for v, w in zip(row, widths)]
            tall = max(row_height, max(map(len, blocks))*size*1.3/72+.13)
            if i % 2 == 0:
                box(fig, left, y, usable, tall)
            x = left
            for j, (block, span) in enumerate(zip(blocks, widths)):
                ha, px = ("left", x+.065) if j < 2 else ("right", x+span-.065)
                for k, value in enumerate(block):
                    strong = highlight is not None and (i, j) in highlight
                    text(fig, px, y+.065+k*size*1.3/72, value, size,
                        teal if strong else ink, "bold" if strong else "normal", ha)
                x += span
            y += tall
        return y

    def finish(fig, pdf, number):
        pdf.savefig(fig)
        fig.savefig(output_dir/f"report_page_{number:02d}.png", dpi=120)
        plt.close(fig)

    with PdfPages(pdf_path, metadata={"Title": "Improving Routing Accuracy: UCB, KL-UCB and LCB",
            "Subject": "Offline paired comparison on 3000 questions and 10000 resampled rounds"}) as pdf:
        fig = base(1, "Testing three confidence-bound changes",
            "Fixed-threshold routing on the same 12,000 saved model outcomes, with the original FT-UCB as baseline.")
        box(fig, left, 1.99, usable, .79)
        text(fig, left+.13, 2.12,
            f"{summary['runs']:,} replays · {summary['decisions']:,} decisions · network attempts: 0", 12, teal, "bold")
        para(fig, 2.45, "μ₀ remains an input to the candidate-set decision at every round. It is not a final answer label.",
            9.2, x=left+.13, span=usable-.26)
        section(fig, 3.06, "Methods compared")
        y = para(fig, 3.42, "Original UCB uses mean + sqrt(2 ln(t)/n). Tight UCB changes 2 to 0.5. "
            "KL-UCB uses a Bernoulli KL upper bound, with c=0 as the practical setting and c=3 as a sensitivity check. "
            "LCB changes the original plus sign to minus, retaining coefficient 2.", 9.5)
        para(fig, y+.15, "All methods keep one initial pull per arm, choose the cheapest eligible arm, "
            "and fall back to uniform sampling if no arm qualifies. Only the selected outcome updates the policy.", 9.5)
        section(fig, 5.10, "Highest observed mean accuracy at each threshold")
        rows = []
        winners = []
        for horizon in horizons:
            for threshold in thresholds:
                winner = max((stats[horizon, threshold, p] for p in policy_ids), key=lambda r: r['accuracy_mean'])
                winners.append(winner)
                rows.append([f"{horizon:,}", f"{threshold:.1%}", POLICIES[winner['policy']]['label'],
                    f"{winner['accuracy_mean']:.2%}", f"{winner['paired_accuracy_gain_pp']:+.2f}",
                    f"${winner['fixed_cost_usd_mean']/horizon*1000:.3f}"])
        table(fig, 5.47, ["Rounds", "μ₀", "Highest-mean method", "Accuracy", "Gain (pp)", "Fixed $/1k"],
            rows, [.76, .59, 2.22, 1.17, 1.09, 1.20], size=8.1, row_height=.35)
        para(fig, 9.18, "Gain is the paired difference from original UCB at the same μ₀ and horizon, averaged over "
            f"{summary['repeats']} seeds. A winner in this simulation is not evidence of superiority on unseen questions.",
            8.7, muted)
        para(fig, 10.02, "The 3,000-round experiment uses each source question once. The 10,000-round experiment "
            "samples that same 3,000-question pool with replacement; it does not contain 10,000 new questions.", 9.2)
        finish(fig, pdf, 1)

        for number, horizon in enumerate(horizons, 2):
            sampling = next(r['sampling'] for r in aggregates if r['horizon'] == horizon)
            fig = base(number, f"{horizon:,} rounds: accuracy and cost",
                "One shuffled pass through the source questions." if sampling == 'without_replacement' else
                "Resampling the same source questions with replacement; repeated question IDs are intentional.")
            rows, high = [], set()
            for threshold in thresholds:
                best = max(stats[horizon, threshold, p]['accuracy_mean'] for p in policy_ids)
                for policy in policy_ids:
                    item = stats[horizon, threshold, policy]
                    idx = len(rows)
                    rows.append([f"{threshold:.1%}", POLICIES[policy]['label'],
                        f"{item['accuracy_mean']*100:.2f} ± {item['accuracy_std']*100:.2f}",
                        f"{item['paired_accuracy_gain_pp']:+.2f}",
                        f"{item['fixed_cost_usd_mean']/horizon*1000:.3f}",
                        f"{item['quality_regret_mean']:.2f}", f"{item['fallback_fraction_mean']:.1%}"])
                    if item['accuracy_mean'] == best:
                        high.add((idx, 2))
            end = table(fig, 2.01, ["μ₀", "Method", "Accuracy %\nmean ± SD", "Gain\npp", "Fixed\n$/1k", "Quality\nregret", "Uniform\nfallback"],
                rows, [.57, 1.57, 1.25, .65, .92, 1.07, 1.00], size=7.9, row_height=.315, highlight=high)
            para(fig, max(end+.24, 9.15), "Fixed cost uses the same per-model mean known charge for every method. "
                "It is a comparable policy-cost estimate, not an invoice total. Selected missing charges remain missing "
                "in runs.csv; known actual charges are reported separately there.", 8.6, muted)
            para(fig, 10.14, "SD measures order/resampling variation across paired seeds, conditional on the supplied labels. "
                "Quality regret uses per-arm positive gaps below μ₀. Uniform fallback is the fraction of all rounds "
                "with an empty candidate set.", 8.6, muted)
            finish(fig, pdf, number)

        number = len(horizons)+2
        fig = base(number, f"Learning curves at μ₀ = {focus_threshold:.1%}",
            "Mean paths over matched seeds; cost and quality regret retain the original positive-part definitions.")
        axes = fig.subplots(3, len(horizons), squeeze=False)
        fig.subplots_adjust(left=.11, right=.96, bottom=.19, top=.81, wspace=.36, hspace=.49)
        for column, horizon in enumerate(horizons):
            for policy in policy_ids:
                path = sorted((r for r in curves if r['horizon'] == horizon and
                    r['threshold'] == focus_threshold and r['policy'] == policy), key=lambda r: r['step'])
                for row, field in enumerate(('accuracy', 'cost_regret_usd', 'quality_regret')):
                    scale = 100 if field == 'accuracy' else 1
                    axes[row, column].plot([r['step'] for r in path], [r[field+'_mean']*scale for r in path],
                        color=COLORS[policy], linewidth=1.5, label=POLICIES[policy]['label'])
            axes[0, column].set_title(f"{horizon:,} rounds", fontsize=10, color=ink)
            for row, ylabel in enumerate(('Accuracy (%)', 'Cost regret (USD)', 'Quality regret')):
                ax = axes[row, column]
                ax.set_xlabel('Round t', fontsize=7.5)
                ax.set_ylabel(ylabel, fontsize=7.5)
                ax.tick_params(labelsize=7)
                ax.grid(color=line, linewidth=.6)
                ax.set_xlim(1, horizon)
                ax.set_ylim(bottom=65 if row == 0 else 0)
                if row == 0:
                    ax.set_ylim(65, 100)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.53, .135), ncol=3, frameon=False, fontsize=8)
        para(fig, 10.30, "The plotted curves are sampled every 50 rounds, with initialization and terminal rounds included. "
            "Every primary-seed decision, candidate set and selected confidence bound is retained in primary_trace.csv.", 8.7, muted)
        finish(fig, pdf, number)

        number += 1
        fig = base(number, "Selection behavior and paired gains",
            f"The first seed ({summary['primary_seed']}) is shown for auditability; aggregate comparisons use all paired seeds.")
        rows = []
        name_by_substring = ('haiku', 'luna', 'gemini', 'sonnet')
        for horizon in horizons:
            for policy in policy_ids:
                run = next(r for r in runs if r['horizon'] == horizon and r['threshold'] == focus_threshold and
                    r['policy'] == policy and r['seed'] == summary['primary_seed'])
                counts = run['model_selections']
                values = [next(v for k, v in counts.items() if word in k) for word in name_by_substring]
                rows.append([f"{horizon:,}", POLICIES[policy]['label'], *[f"{v:,}" for v in values],
                    f"{run['fallback_fraction']:.1%}"])
        end = table(fig, 2.01, ["Rounds", "Method", "Haiku", "Luna", "Gemini", "Sonnet", "Fallback"],
            rows, [.70, 2.02, .82, .82, .87, .87, .93], size=8, row_height=.36)
        section(fig, end+.36, "Why an LCB does not always select the best model")
        y = para(fig, end+.72, "A sign change makes eligibility more conservative. Early lower bounds can all fall "
            "below μ₀, so the unmodified fallback chooses uniformly. Even when candidates exist, the rule selects "
            "the cheapest candidate, rather than the model with the largest mean. A small margin between a model’s "
            "accuracy and μ₀ can keep its LCB below the threshold for many rounds.", 9.3)
        section(fig, y+.30, "Paired accuracy gain for the highest-mean method")
        rows = []
        for horizon in horizons:
            item = max((stats[horizon, focus_threshold, p] for p in policy_ids), key=lambda r: r['accuracy_mean'])
            low, high = item['gain_mc_ci95_low_pp'], item['gain_mc_ci95_high_pp']
            interval = f"{low:+.2f} to {high:+.2f}" if low is not None else 'one seed only'
            rows.append([f"{horizon:,}", POLICIES[item['policy']]['label'],
                f"{item['paired_accuracy_gain_pp']:+.2f} pp", interval])
        table(fig, y+.66, ['Rounds', 'Method at highest μ₀', 'Mean gain', '95% MC interval (pp)'],
            rows, [.76, 2.20, 1.48, 2.59], size=8.3, row_height=.40)
        para(fig, 10.18, "The interval is a normal approximation for the simulation mean using paired seed differences. "
            "It is not a confidence interval for performance on unseen questions. The full threshold-by-method results "
            "are available in aggregates.csv.", 8.6, muted)
        finish(fig, pdf, number)

        number += 1
        fig = base(number, "Definitions, sources and limits",
            "The three requested ideas are evaluated empirically; the experiments do not establish new regret theorems.")
        section(fig, 1.99, "Confidence bounds and threshold decisions")
        y = para(fig, 2.34, "UCB: mean + sqrt(a ln(t)/n), with a=2 for the original and a=0.5 for the tighter variant. "
            "LCB: mean − sqrt(2 ln(t)/n). Bernoulli KL-UCB: the largest q ≥ mean satisfying "
            "n·KL(mean,q) ≤ ln(t)+c ln(ln(t)), using c=0 and c=3. For KL policies, testing the inequality at μ₀ "
            "gives the same candidate membership as explicitly solving the upper bound.", 9.1)
        y = para(fig, y+.16, "Each policy forms candidates with bound ≥ μ₀, then chooses the lowest fixed cost. "
            "μ₀ is supplied before the run. It does not define the 0/1 correctness labels, and there is no final "
            "accuracy filter. A model with lower true accuracy may remain eligible while its bound is wide.", 9.1)
        section(fig, y+.29, "Regrets and cost accounting")
        y = para(fig, y+.64, "Cost regret is Σ max(0, c[selected]−c[a*]), where a* is the cheapest arm whose "
            "full-data mean accuracy reaches μ₀. Quality regret is Σ max(0, μ₀−μ[selected]). Full-data means "
            "are used only for evaluation; policy updates see the selected saved reward. Fixed costs are calibrated "
            "from known CSV charges. Unknown charges are never substituted with zero.", 9.1)
        section(fig, y+.29, "Dataset, sampling and reproducibility")
        y = para(fig, y+.64, f"The supplied results_3000.csv contains {summary['questions']:,} questions and "
            f"{summary['outcomes']:,} model outcomes. Verified archived labels are preserved, and the remaining "
            f"answers use the same previous scoring rules. {summary['needs_review']} records retain manual-review flags. "
            "All methods share question streams, seeds, costs and labels. Longer runs resample the same saved pool; "
            "no additional unique questions, predictions or API requests are created.", 9.1)
        y = para(fig, y+.15, f"Seeds start at {summary['primary_seed']} and cover {summary['repeats']} independent "
            "order/resampling realizations. Their SD and paired intervals quantify Monte Carlo variability for "
            "this fixed pool. They do not measure generalization uncertainty or remove the scoring-review limitation.", 9.1)
        section(fig, y+.29, "Theory and source attribution")
        y = para(fig, y+.64, "Garivier and Cappé recommend c=0 for practical KL-UCB experiments and analyze c=3 "
            "for standard reward-maximizing KL-UCB. Their refined quadratic bound also uses a log-log correction. "
            "These results motivate the variants; they do not automatically prove the same guarantees for this "
            "cost-minimizing threshold policy or its LCB replacement. The original FT-UCB theorem is not re-proved here.", 8.9)
        text(fig, left, min(y+.22, 10.47), 'Original FT-UCB: Juneja et al., Algorithm 6, Appendix H.',
            8, teal, url=PAPER)
        text(fig, left, min(y+.44, 10.69), 'Bernoulli KL-UCB: Garivier & Cappé (2011), Algorithms and Proposition 4.',
            8, teal, url=KL_PAPER)
        finish(fig, pdf, number)

    metadata = {"pdf": str(pdf_path), "pages": page_count,
                "highest_mean_cases": [{k: winner[k] for k in ('horizon', 'threshold', 'policy',
                    'accuracy_mean', 'paired_accuracy_gain_pp')} for winner in winners]}
    (output_dir/'report_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
