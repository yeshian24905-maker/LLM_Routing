"""Paired offline comparison of domain-aware primal-dual UCB and Thompson routing with FT-UCB baselines."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from .compare import network_disabled, prompt_stream, write_csv
from .confidence import FixedThresholdConfidence
from .data import load_csv_replay
from .domain_dual import DomainPrimalDualUCB
from .domain_oracle import oracle_routing
from .domain_thompson import DomainPrimalDualThompson
from .domain_plot import plot_tradeoff
from .ft_evaluate import DEFAULT_THRESHOLDS, arm_statistics, resolve_costs


class GlobalFT:
    """One FT policy for all questions; the domain is ignored."""

    def __init__(self, models, costs, threshold, kind, seed):
        self.policy = FixedThresholdConfidence(models, costs, threshold, kind, seed=seed)

    def select(self, domain):
        return self.policy.select(diagnostics=False)

    def update(self, domain, arm, correct):
        self.policy.update(arm, correct)


class PerDomainFT:
    """An independent FT policy per domain, each with the same threshold."""

    def __init__(self, models, costs, threshold, kind, seed):
        self.args = models, costs, threshold, kind
        self.seed = seed
        self.policies = {}

    def _policy(self, domain):
        if domain not in self.policies:
            models, costs, threshold, kind = self.args
            self.policies[domain] = FixedThresholdConfidence(
                models, costs, threshold, kind, seed=f"{self.seed}:{domain}")
        return self.policies[domain]

    def select(self, domain):
        return self._policy(domain).select(diagnostics=False)

    def update(self, domain, arm, correct):
        self._policy(domain).update(arm, correct)


def policy_factories(kappas, etas, betas, lambda0):
    """Map each policy id to a factory(models, costs, threshold, seed)."""
    factories = {
        "ft_ucb": lambda m, c, t, s: GlobalFT(m, c, t, "ucb_original", s),
        "ft_kl_ucb": lambda m, c, t, s: GlobalFT(m, c, t, "kl_ucb", s),
        "ft_kl_ucb_per_domain": lambda m, c, t, s: PerDomainFT(m, c, t, "kl_ucb", s),
    }
    for kappa in kappas:
        tag = "inf" if math.isinf(kappa) else f"{kappa:g}"
        for eta in etas:
            for beta in betas:
                factories[f"pd_k{tag}_eta{eta:g}_beta{beta:g}"] = (lambda k, e, b: lambda m, c, t, s: DomainPrimalDualUCB(
                    m, c, t, kappa=k, beta=b, eta=e, lambda0=lambda0))(kappa, eta, beta)
            factories[f"ts_k{tag}_eta{eta:g}"] = (lambda k, e: lambda m, c, t, s: DomainPrimalDualThompson(
                m, c, t, kappa=k, eta=e, lambda0=lambda0, seed=f"ts:{s}"))(kappa, eta)
    return factories


def _moments(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def run_domain_comparison(table, domains, models, costs, factories, *, horizons, thresholds,
                          repeats=30, seed=42, checkpoint=100, progress=None):
    """Replay every policy on the same shuffled stream per seed; return runs, aggregates, curves.

    Policies see only the domain of each question and the saved outcome of the
    model they pick. Costs are the fixed per-model costs, in USD per 1000 questions.
    """
    runs, checkpoints = [], {}
    for horizon in horizons:
        for run_seed in range(seed, seed + repeats):
            stream, sampling = prompt_stream(table, horizon, run_seed)
            for threshold in thresholds:
                for policy_id, factory in factories.items():
                    policy = factory(models, costs, threshold, f"ft:{run_seed}")
                    correct, spent = 0, 0.0
                    picks = dict.fromkeys(models, 0)
                    for t, question_id in enumerate(stream, 1):
                        domain = domains[question_id]
                        arm = policy.select(domain)
                        outcome = table[question_id][arm].correct
                        policy.update(domain, arm, outcome)
                        correct += outcome
                        spent += costs[arm]
                        picks[arm] += 1
                        if t % checkpoint == 0 or t == horizon:
                            checkpoints.setdefault((horizon, threshold, policy_id, t), []).append(
                                (correct / t, spent / t * 1000))
                    runs.append({
                        "horizon": horizon, "sampling": sampling, "threshold": threshold,
                        "policy": policy_id, "seed": run_seed, "accuracy": correct / horizon,
                        "cost_usd_per_1k": spent / horizon * 1000,
                        "meets_threshold": int(correct / horizon >= threshold),
                        "final_accuracy_price": getattr(policy, "accuracy_price", None),
                        "model_share": {m: picks[m] / horizon for m in models},
                    })
            if progress:
                progress(f"horizon={horizon} seed={run_seed} ({run_seed - seed + 1}/{repeats})")
    oracles = {}
    for threshold in thresholds:
        for scope, per_domain in (("global", False), ("domain", True)):
            result = oracle_routing(table, domains, models, costs, threshold, per_domain)
            oracles[threshold, scope] = None if result["cost_usd"] is None else result["cost_usd"] * 1000
    aggregates = []
    for horizon in horizons:
        for threshold in thresholds:
            for policy_id in factories:
                group = [r for r in runs if (r["horizon"], r["threshold"], r["policy"]) ==
                         (horizon, threshold, policy_id)]
                acc, acc_sd = _moments([r["accuracy"] for r in group])
                cost, cost_sd = _moments([r["cost_usd_per_1k"] for r in group])
                aggregates.append({
                    "horizon": horizon, "threshold": threshold, "policy": policy_id,
                    "repeats": len(group), "accuracy_mean": acc, "accuracy_std": acc_sd,
                    "cost_usd_per_1k_mean": cost, "cost_usd_per_1k_std": cost_sd,
                    "meets_threshold_rate": statistics.mean(r["meets_threshold"] for r in group),
                    "oracle_global_cost_usd_per_1k": oracles[threshold, "global"],
                    "oracle_domain_cost_usd_per_1k": oracles[threshold, "domain"],
                    "model_share_mean": {m: statistics.mean(r["model_share"][m] for r in group)
                                         for m in models},
                })
    curves = []
    for (horizon, threshold, policy_id, t), values in sorted(checkpoints.items()):
        acc, acc_sd = _moments([v[0] for v in values])
        cost, cost_sd = _moments([v[1] for v in values])
        curves.append({"horizon": horizon, "threshold": threshold, "policy": policy_id, "step": t,
                       "accuracy_mean": acc, "accuracy_std": acc_sd,
                       "cost_usd_per_1k_mean": cost, "cost_usd_per_1k_std": cost_sd})
    return runs, aggregates, curves


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, default=root / "results_3000.csv")
    parser.add_argument("--scores-csv", type=Path, default=root / "bandit/output/ft-ucb/scored_results.csv")
    parser.add_argument("--expected-prompts", type=int, default=3000)
    parser.add_argument("--horizons", type=int, nargs="+", default=[3000, 10000])
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kappas", type=float, nargs="+", default=[math.inf, 5, 20, 50],
                        help="Prior weights in pseudo-observations; inf ignores domains.")
    parser.add_argument("--etas", type=float, nargs="+", default=[1.0])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.5])
    parser.add_argument("--lambda0", type=float, default=0.0)
    parser.add_argument("--no-baselines", action="store_true", help="Run only the primal-dual UCB and Thompson variants.")
    parser.add_argument("--output-dir", type=Path, default=root / "bandit/output/domain-comparison")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--empirical-costs", action="store_true")
    source.add_argument("--arm-costs", type=Path)
    args = parser.parse_args(argv)
    with network_disabled() as attempts:
        source_hash = hashlib.sha256(args.results_csv.read_bytes()).hexdigest()
        table, models, scored = load_csv_replay(args.results_csv, args.scores_csv, args.expected_prompts)
        domains = {row["question_id"]: row["subject"] for row in scored}
        stats = arm_statistics(table, models)
        fixed = json.loads(args.arm_costs.read_text(encoding="utf-8-sig")) if args.arm_costs else None
        costs = resolve_costs(models, stats, fixed, args.empirical_costs)
        factories = policy_factories(args.kappas, args.etas, args.betas, args.lambda0)
        if args.no_baselines:
            factories = {k: v for k, v in factories.items() if k.startswith(("pd_", "ts_"))}
        runs, aggregates, curves = run_domain_comparison(
            table, domains, models, costs, factories, horizons=args.horizons,
            thresholds=args.thresholds, repeats=args.repeats, seed=args.seed,
            progress=lambda s: print(s, flush=True))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in (("runs", runs), ("aggregates", aggregates), ("curves", curves)):
            write_csv(args.output_dir / f"{name}.csv", rows)
        summary = {
            "questions": len(table), "domains": len(set(domains.values())), "models": models,
            "arm_costs_usd": costs, "arm_statistics": stats, "policies": list(factories),
            "kappas": [str(k) for k in args.kappas], "etas": args.etas, "betas": args.betas,
            "lambda0": args.lambda0, "horizons": args.horizons, "thresholds": args.thresholds,
            "repeats": args.repeats, "primary_seed": args.seed, "input_sha256": source_hash,
            "network_connections_attempted": len(attempts),
            "oracle_note": "LP optimum from all saved outcomes; evaluation only, never seen by a policy.",
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        plot_tradeoff(aggregates, args.output_dir / "tradeoff.png",
                      [t for t in (0.825, 0.85, 0.89) if t in args.thresholds] or args.thresholds)
    print(f"Complete: {len(runs)} runs; network attempts: {len(attempts)}; output: {args.output_dir}")


if __name__ == "__main__":
    main()
