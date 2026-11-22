"""Reproduce the 3,000-question analysis using saved local responses only.

Run from any directory: python -B analysis_3000/analyze.py
No API client is imported and no input file is written.
"""

import csv
import hashlib
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, fisher_exact

from scoring import SCORER, grade


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
DATA = ROOT / "mmmu_test_sample"
REFERENCE = Path(r"C:\Users\24905\Downloads\Agentic Routing Results.docx")
MODELS = ["anthropic/claude-haiku-5.5", "openai/gpt-5.6-luna",
          "google/gemini-3.8-flash", "anthropic/claude-sonnet-5.5"]
NAMES = dict(zip(MODELS, ["Haiku", "Luna", "Gemini", "Sonnet"]))
COLORS = ["#278779", "#df9b35", "#4278b6", "#9b65a7"]
THRESHOLDS = [.89, .85, .80, .825]
REPEATS = 100
SEED = 42


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                      allow_nan=False), encoding="utf-8")


def table(name, rows):
    path = OUT / "tables" / f"{name}.csv"
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def wilson(k, n):
    if not n:
        return None, None
    z = 1.959963984540054
    p = k / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return center-half, center+half


def holm(rows, key="p_value"):
    adjusted = 0.
    for rank, row in enumerate(sorted(rows, key=lambda r: r[key])):
        adjusted = max(adjusted, min(1., row[key] * (len(rows)-rank)))
        row["p_holm"] = adjusted


def finite_metric(s):
    if s == "":
        return None
    x = float(s)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"Invalid nonnegative metric: {s}")
    return x


def metrics(rows):
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    lo, hi = wilson(correct, n)
    known = [r["cost_value"] for r in rows if r["cost_value"] is not None]
    effective = [r["effective_cost"] for r in rows]
    if any(x is None for x in effective):
        raise ValueError("Unresolved cost; do not silently impute zero")
    cost = math.fsum(effective)
    latency = [r["latency_value"] for r in rows if r["latency_value"] is not None]
    d = {
        "n": n, "correct": correct, "accuracy": correct/n,
        "ci95_low": lo, "ci95_high": hi,
        "known_cost_usd": math.fsum(known), "raw_missing_cost_count": n-len(known),
        "documented_no_charge_count": sum(r["cost_source"] == "provider_no_charge" for r in rows),
        "latest_outcome_cost_usd": cost, "mean_outcome_cost_usd": cost/n,
        "mean_known_cost_usd": math.fsum(known)/len(known),
        "cost_per_correct_usd": cost/correct if correct else None,
        "request_errors": sum(bool(r["error"]) for r in rows),
        "empty_responses": sum(not r["raw_response"].strip() for r in rows),
        "content_filter_count": sum(r["finish_reason"] == "content_filter" for r in rows),
        "length_count": sum(r["finish_reason"] == "length" for r in rows),
        "needs_review": sum(r["needs_review"] for r in rows),
        "latency_n": len(latency),
        "latency_mean_s": float(np.mean(latency)),
        "latency_p50_s": float(np.quantile(latency, .5)),
        "latency_p95_s": float(np.quantile(latency, .95)),
    }
    for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
        values = [finite_metric(r[field]) for r in rows if r[field] != ""]
        d[f"{field}_n"] = len(values)
        d[f"{field}_mean"] = float(np.mean(values)) if values else None
    return d


def savefig(name, fig):
    fig.savefig(OUT / "figures" / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "figures" / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def replay(order, matrix, actual_cost, costs, means, threshold, seed, keep=False):
    """Literal FT-UCB: choose first, then reveal ONLY the selected reward."""
    counts = np.zeros(4, dtype=int)
    wins = np.zeros(4, dtype=int)
    rng = random.Random(f"ft:{seed}")
    feasible = [i for i in range(4) if means[i] >= threshold]
    optimal = feasible[0] if feasible else None
    total_wins = 0
    total_cost = fixed_cost = rc = rq = 0.
    min_haiku_ucb = float("inf")
    first_after_init = None
    path = []
    for t, q in enumerate(order, 1):
        bounds = None
        if t <= 4:
            a = t-1
            phase = "initialization"
        else:
            # A four-element Python loop is faster than repeated NumPy allocations.
            bounds = [wins[i]/counts[i] + math.sqrt(2*math.log(t)/counts[i]) for i in range(4)]
            candidates = [i for i, u in enumerate(bounds) if u >= threshold]
            a = candidates[0] if candidates else rng.randrange(4)
            phase = "feasible_min_cost" if candidates else "uniform_fallback"
            min_haiku_ucb = min(min_haiku_ucb, bounds[0])
            if a != 0 and first_after_init is None:
                first_after_init = t
        prior_n = int(counts[a])
        prior_mean = float(wins[a]/prior_n) if prior_n else None
        y = int(matrix[q, a])
        counts[a] += 1
        wins[a] += y
        total_wins += y
        total_cost += actual_cost[q, a]
        fixed_cost += costs[a]
        rc += max(0., costs[a]-costs[optimal]) if optimal is not None else 0.
        rq += max(0., threshold-means[a])
        if keep:
            path.append({
                "threshold": threshold, "seed": seed, "round": t,
                "question_index": q, "model": MODELS[a], "phase": phase,
                "prior_arm_count": prior_n, "prior_arm_mean": prior_mean,
                "selected_ucb": bounds[a] if bounds else None,
                "haiku_ucb": bounds[0] if bounds else None,
                "correct": y, "observed_cost_usd": actual_cost[q, a],
                "cumulative_correct": total_wins, "cumulative_accuracy": total_wins/t,
                "cumulative_observed_cost_usd": total_cost,
                "cumulative_fixed_cost_usd": fixed_cost,
                "cost_regret_usd": rc if optimal is not None else None,
                "quality_regret": rq,
            })
    result = {
        "threshold": threshold, "seed": seed, "rounds": len(order),
        "correct": total_wins, "accuracy": total_wins/len(order),
        "observed_cost_usd": total_cost, "fixed_cost_usd": fixed_cost,
        "cost_regret_usd": rc if optimal is not None else None,
        "quality_regret": rq, "meets_threshold": int(total_wins/len(order) >= threshold),
        "optimal_feasible_model": MODELS[optimal] if optimal is not None else None,
        "haiku_count": int(counts[0]), "luna_count": int(counts[1]),
        "gemini_count": int(counts[2]), "sonnet_count": int(counts[3]),
        "min_haiku_ucb_after_initialization": min_haiku_ucb,
        "first_non_haiku_round_after_initialization": first_after_init,
    }
    return result, path


def main():
    for directory in ("tables", "figures"):
        (OUT / directory).mkdir(exist_ok=True)
    source_paths = [DATA / "result.csv", REFERENCE]
    for base in [DATA, DATA / "additional_2000"]:
        source_paths += [base / "questions.jsonl", base / "dataset_manifest.json",
                         base / "runs/sequential/results.csv", base / "runs/sequential/attempts.jsonl",
                         base / "runs/sequential/run_manifest.json", base / "runs/sequential/summary.json"]
    before = {str(p): sha(p) for p in source_paths}
    questions, cohorts = {}, {}
    image_count = 0
    fingerprints = set()
    for base, label, expected in [(DATA, "original_1000", 1000),
                                  (DATA / "additional_2000", "additional_2000", 2000)]:
        manifest = json.loads((base / "dataset_manifest.json").read_text(encoding="utf-8-sig"))
        assert sha(base / "questions.jsonl") == manifest["questions_sha256"]
        items = [json.loads(s) for s in (base / "questions.jsonl").read_text(encoding="utf-8-sig").splitlines() if s.strip()]
        assert len(items) == expected
        for q in items:
            qid = q["question_id"]
            assert qid not in questions
            content = json.dumps([q["subject"], q["question_type"], q["question"],
                                  q["options"], sorted(q["image_sha256"].items())], sort_keys=True)
            assert content not in fingerprints, f"Duplicate prompt content: {qid}"
            fingerprints.add(content)
            for index, rel in q["images"].items():
                p = (base / rel).resolve()
                assert p.is_relative_to(base.resolve()) and sha(p) == q["image_sha256"][index]
                image_count += 1
            questions[qid], cohorts[qid] = q, label
    qids = list(questions)
    assert len(qids) == 3000
    assert set(Counter(q["subject"] for q in questions.values()).values()) == {100}
    assert len(set(q["subject"] for q in questions.values())) == 30
    rows = read_csv(DATA / "result.csv")
    originals = read_csv(DATA / "runs/sequential/results.csv")
    additions = read_csv(DATA / "additional_2000/runs/sequential/results.csv")
    assert rows == originals + additions
    assert len(rows) == 12000 and len({(r["question_id"], r["model"]) for r in rows}) == 12000
    assert Counter(r["model"] for r in rows) == Counter({m: 3000 for m in MODELS})
    by_key = {}
    score_rows = []
    for r in rows:
        q = questions[r["question_id"]]
        assert all(r[k] == q[k] for k in ("subject", "question_type", "correct_answer"))
        prediction, extraction, correct = grade(r, q["options"])
        r.update(extracted_answer=prediction, extraction=extraction, correct=correct,
                 cohort=cohorts[r["question_id"]], difficulty=q["topic_difficulty"],
                 image_count=len(q["images"]))
        r["needs_review"] = int(q["question_type"] == "open" or extraction not in (
            "answer_tag", "empty_response", "request_error"))
        r["cost_value"] = finite_metric(r["cost"])
        r["latency_value"] = finite_metric(r["latency"])
        r["cost_source"] = "recorded" if r["cost_value"] is not None else (
            "provider_no_charge" if r["error"].startswith("moderation_error:")
            and "No credits were charged" in r["error"] else "unresolved")
        r["effective_cost"] = r["cost_value"] if r["cost_value"] is not None else (
            0. if r["cost_source"] == "provider_no_charge" else None)
        assert r["effective_cost"] is not None
        by_key[r["question_id"], r["model"]] = r
        score_rows.append({k: r[k] for k in (
            "question_id", "subject", "question_type", "difficulty", "image_count", "cohort", "model",
            "correct_answer", "extracted_answer", "extraction", "correct", "needs_review", "finish_reason",
            "cost", "cost_source", "effective_cost")})
    assert all((qid, m) in by_key for qid in qids for m in MODELS)
    table("scored_results", score_rows)
    table("review_queue", [r for r in score_rows if r["needs_review"]])
    table("response_issues", [{"question_id": r["question_id"], "model": r["model"],
                               "cohort": r["cohort"], "subject": r["subject"],
                               "issue": "moderation_error" if r["error"] else r["finish_reason"],
                               "cost_source": r["cost_source"], "cost": r["cost"],
                               "correct": r["correct"]}
                              for r in rows if r["error"] or not r["raw_response"].strip()
                              or r["finish_reason"] == "length"])

    summaries = [{"model": m, **metrics([r for r in rows if r["model"] == m])} for m in MODELS]
    table("model_summary", summaries)
    groups = {}
    for field in ("subject", "question_type", "difficulty", "image_count", "cohort"):
        a = []
        for v in sorted({r[field] for r in rows}, key=str):
            for m in MODELS:
                a.append({field: v, "model": m, **metrics([r for r in rows if r[field] == v and r["model"] == m])})
        table(f"by_{field}", a)
        groups[field] = a

    matrix = np.array([[by_key[q, m]["correct"] for m in MODELS] for q in qids], dtype=int)
    actual_cost = np.array([[by_key[q, m]["effective_cost"] for m in MODELS] for q in qids])
    means = matrix.mean(axis=0)
    costs = np.array([r["mean_known_cost_usd"] for r in summaries])
    assert list(np.argsort(costs)) == list(range(4)), "FT-UCB warm-start arm order changed"
    pairs = []
    for i, j in itertools.combinations(range(4), 2):
        x, y = matrix[:, i], matrix[:, j]
        aonly, bonly = int(np.sum((x==1)&(y==0))), int(np.sum((x==0)&(y==1)))
        both_wrong = int(np.sum((x==0)&(y==0)))
        pairs.append({
            "model_a": MODELS[i], "model_b": MODELS[j], "a_only_correct": aonly,
            "b_only_correct": bonly, "both_correct": int(np.sum((x==1)&(y==1))),
            "both_wrong": both_wrong, "b_minus_a_accuracy_pp": 100*(float(y.mean()-x.mean())),
            "binary_correctness_agreement": float(np.mean(x==y)),
            "pair_oracle_accuracy": float(np.mean((x+y)>0)),
            "b_rescues_fraction_of_a_errors": bonly/int(np.sum(x==0)),
            "a_rescues_fraction_of_b_errors": aonly/int(np.sum(y==0)),
            "p_value": float(binomtest(aonly, aonly+bonly, .5).pvalue) if aonly+bonly else 1.,
        })
    holm(pairs)
    table("paired_comparisons", pairs)
    cohort_changes = []
    baseline = {"Haiku": 824, "Luna": 827, "Gemini": 908, "Sonnet": 879}
    for m in MODELS:
        a, b = [[r for r in rows if r["model"] == m and r["cohort"] == g]
                for g in ("original_1000", "additional_2000")]
        k1, k2 = sum(r["correct"] for r in a), sum(r["correct"] for r in b)
        p1, p2 = k1/len(a), k2/len(b)
        l1, u1 = wilson(k1, len(a)); l2, u2 = wilson(k2, len(b))
        diff = p2-p1
        cohort_changes.append({
            "model": m, "old_correct_reference_word": baseline[NAMES[m]],
            "old_correct_regraded": k1, "new_correct": k2,
            "old_accuracy": p1, "new_accuracy": p2,
            "new_minus_old_pp": 100*diff,
            "difference_ci95_low_pp": 100*(diff-math.sqrt((p2-l2)**2+(u1-p1)**2)),
            "difference_ci95_high_pp": 100*(diff+math.sqrt((u2-p2)**2+(p1-l1)**2)),
            "p_value": float(fisher_exact([[k1, len(a)-k1], [k2, len(b)-k2]]).pvalue),
        })
    holm(cohort_changes)
    table("cohort_comparison", cohort_changes)

    number_correct = matrix.sum(axis=1)
    overlap = [{"correct_models": k, "questions": int(np.sum(number_correct==k)),
                "fraction": float(np.mean(number_correct==k))} for k in range(5)]
    table("correctness_overlap", overlap)
    question_analysis = []
    oracle_rows = []
    for q, qid in enumerate(qids):
        eligible = np.flatnonzero(matrix[q]).tolist() or list(range(4))
        fixed_arm = eligible[0]
        # Lowest actual per-question outcome cost, tie broken by fixed arm order.
        actual_arm = min(eligible, key=lambda i: (actual_cost[q, i], i))
        qrow = {"question_id": qid, "subject": questions[qid]["subject"],
                "question_type": questions[qid]["question_type"], "cohort": cohorts[qid],
                "difficulty": questions[qid]["topic_difficulty"], "correct_models": int(number_correct[q])}
        qrow.update({f"{NAMES[m].lower()}_correct": int(matrix[q, i]) for i, m in enumerate(MODELS)})
        qrow.update(oracle_fixed_order_model=MODELS[fixed_arm],
                    oracle_fixed_order_cost_usd=float(actual_cost[q, fixed_arm]),
                    oracle_actual_min_cost_model=MODELS[actual_arm],
                    oracle_actual_min_cost_usd=float(actual_cost[q, actual_arm]))
        question_analysis.append(qrow)
    table("question_outcomes", question_analysis)
    table("all_models_wrong", [r for r in question_analysis if r["correct_models"] == 0])
    table("only_one_model_correct", [r for r in question_analysis if r["correct_models"] == 1])
    for label, model_key, cost_key in [
        ("fixed_mean_cost_order", "oracle_fixed_order_model", "oracle_fixed_order_cost_usd"),
        ("actual_per_question_min_cost", "oracle_actual_min_cost_model", "oracle_actual_min_cost_usd")]:
        oracle_rows.append({"oracle": label, "correct": int(np.sum(number_correct>0)),
                            "accuracy": float(np.mean(number_correct>0)),
                            "cost_usd": math.fsum(r[cost_key] for r in question_analysis),
                            **{f"{NAMES[m].lower()}_count": sum(r[model_key]==m for r in question_analysis) for m in MODELS}})
    table("oracles", oracle_rows)

    static = []
    for mu in THRESHOLDS:
        feasible = [i for i in range(4) if means[i] >= mu]
        a = feasible[0] if feasible else None
        static.append({"threshold": mu, "feasible_models": "; ".join(MODELS[i] for i in feasible),
                       "cheapest_feasible_model": MODELS[a] if a is not None else "",
                       "model_accuracy": float(means[a]) if a is not None else None,
                       "observed_total_cost_usd": summaries[a]["latest_outcome_cost_usd"] if a is not None else None,
                       "ci95_low": summaries[a]["ci95_low"] if a is not None else None,
                       "ci95_low_meets_threshold": int(summaries[a]["ci95_low"] >= mu) if a is not None else 0})
    table("static_thresholds", static)

    runs, traces = [], []
    orders = []
    for seed in range(SEED, SEED+REPEATS):
        order = list(range(3000))
        random.Random(f"order:{seed}").shuffle(order)
        orders.append(order)
    for threshold in THRESHOLDS:
        for seed, order in zip(range(SEED, SEED+REPEATS), orders):
            result, path = replay(order, matrix, actual_cost, costs, means, threshold, seed, keep=seed==SEED)
            runs.append(result)
            for row in path:
                row["question_id"] = qids[row.pop("question_index")]
            traces.extend(path)
    table("ft_ucb_runs", runs)
    table("ft_ucb_trace", traces)
    primary = [r for r in runs if r["seed"] == SEED]
    table("ft_ucb_primary", primary)
    sensitivity = []
    for mu in THRESHOLDS:
        a = [r for r in runs if r["threshold"] == mu]
        item = {"threshold": mu, "replays": REPEATS,
                "fraction_meeting_threshold": float(np.mean([r["meets_threshold"] for r in a]))}
        for k in ("accuracy", "observed_cost_usd", "cost_regret_usd", "quality_regret", "haiku_count", "gemini_count"):
            v = [r[k] for r in a]
            item.update({f"{k}_mean": float(np.mean(v)), f"{k}_p05": float(np.quantile(v,.05)),
                         f"{k}_p95": float(np.quantile(v,.95)), f"{k}_min": float(np.min(v)),
                         f"{k}_max": float(np.max(v))})
        sensitivity.append(item)
    table("ft_ucb_order_sensitivity", sensitivity)

    # Literal independent reference for the main replay; no policy state is shared.
    for mu in THRESHOLDS:
        observations = [r for r in traces if r["threshold"] == mu]
        n, s = [0]*4, [0]*4
        fallback = random.Random(f"ft:{SEED}")
        for t, (q, trace) in enumerate(zip(orders[0], observations), 1):
            if t <= 4:
                chosen = t-1
            else:
                candidates = [i for i in range(4) if s[i]/n[i]+math.sqrt(2*math.log(t)/n[i]) >= mu]
                chosen = min(candidates, key=lambda i: costs[i]) if candidates else fallback.choice(list(range(4)))
            assert trace["model"] == MODELS[chosen]
            assert trace["correct"] == matrix[q, chosen]
            n[chosen] += 1; s[chosen] += int(matrix[q, chosen])
        result = next(r for r in primary if r["threshold"] == mu)
        assert result["correct"] == sum(s) and sum(n) == 3000
        assert math.isclose(result["quality_regret"], sum(n[i]*max(0.,mu-means[i]) for i in range(4)), abs_tol=1e-8)
        optimal = next(i for i in range(4) if means[i] >= mu)
        assert math.isclose(result["cost_regret_usd"], sum(n[i]*max(0.,costs[i]-costs[optimal]) for i in range(4)), abs_tol=1e-8)

    # Billing across historical attempts is a separate measure from latest outcomes.
    billing = []
    for base in (DATA, DATA / "additional_2000"):
        costs_by_generation = {}
        unknown_failures = 0
        with (base / "runs/sequential/attempts.jsonl").open(encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f):
                record = json.loads(line)
                if record.get("kind") == "accounting_refresh":
                    costs_by_generation[record["generation_id"]] = record["result"]["cost"]
                    continue
                response = record.get("response")
                generation = response.get("id") if isinstance(response, dict) else None
                error = record["result"]["error"] or ""
                if error.startswith(("api_error", "moderation_error")) and record["result"]["cost"] is None:
                    unknown_failures += 1
                    continue
                costs_by_generation[generation if isinstance(generation,str) else f"line-{line_no}"] = record["result"]["cost"]
        values = list(costs_by_generation.values())
        billing.append({"cohort": "original_1000" if base == DATA else "additional_2000",
                        "completion_requests": len(values), "known_request_cost_usd": math.fsum(x for x in values if x is not None),
                        "unknown_completion_cost_count": sum(x is None for x in values),
                        "failed_requests_without_cost": unknown_failures})
    table("collection_billing", billing)

    # Figures use English labels for consistent rendering on other platforms.
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "axes.titleweight":"bold", "figure.facecolor":"white"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, r in enumerate(summaries):
        axes[0].scatter(i,100*r["accuracy"],color=COLORS[i],s=90,zorder=3)
        axes[0].errorbar(i,100*r["accuracy"],yerr=[[100*(r["accuracy"]-r["ci95_low"])],[100*(r["ci95_high"]-r["accuracy"])]],color="#243746",capsize=4)
        axes[0].text(i,100*r["accuracy"]+1.8,f'{100*r["accuracy"]:.2f}%',ha="center")
        axes[1].bar(i,r["latest_outcome_cost_usd"],color=COLORS[i],width=.65)
        axes[1].text(i,r["latest_outcome_cost_usd"]+.2,f'${r["latest_outcome_cost_usd"]:.4f}',ha="center")
    axes[0].set(ylim=(70,96),ylabel="Accuracy (%)",title="Accuracy with 95% Wilson intervals")
    axes[1].set(ylim=(0,15.5),ylabel="Recorded outcome cost (USD)",title="Cost for 3,000 saved outcomes")
    for ax in axes: ax.set_xticks(range(4),[NAMES[m] for m in MODELS])
    fig.tight_layout(); savefig("overall",fig)

    fig, ax = plt.subplots(figsize=(8,4.4))
    for i,r in enumerate(summaries):
        x = 1000*r["mean_outcome_cost_usd"]
        ax.scatter(x,100*r["accuracy"],s=130,color=COLORS[i])
        ax.annotate(NAMES[MODELS[i]],(x,100*r["accuracy"]),xytext=(7,7),textcoords="offset points")
    ax.set(xlabel="USD per 1,000 questions (latest outcomes)",ylabel="Accuracy (%)",title="Observed cost-quality tradeoff",ylim=(79,92),xlim=(0,5.1))
    ax.grid(alpha=.2);fig.tight_layout();savefig("cost_quality",fig)

    subjects = sorted({q["subject"] for q in questions.values()})
    subject_values = np.array([[next(r["accuracy"] for r in groups["subject"] if r["subject"]==s and r["model"]==m) for m in MODELS] for s in subjects])
    fig,ax=plt.subplots(figsize=(9,12))
    im=ax.imshow(subject_values*100,cmap="YlGnBu",vmin=45,vmax=100,aspect="auto")
    ax.set_xticks(range(4),[NAMES[m] for m in MODELS]);ax.set_yticks(range(30),[s.replace("_"," ") for s in subjects])
    for i in range(30):
        for j in range(4):ax.text(j,i,f"{100*subject_values[i,j]:.0f}%",ha="center",va="center",color="white" if subject_values[i,j]>.8 else "#122f42",fontsize=9)
    ax.set_title("Accuracy by area: 100 questions per row",pad=15)
    fig.colorbar(im,ax=ax,label="Accuracy (%)",fraction=.035,pad=.04)
    fig.tight_layout();savefig("area_accuracy",fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for ax,field,values in [(axes[0],"question_type",["multiple-choice","open"]),(axes[1],"difficulty",["Easy","Medium","Hard"])]:
        for i,m in enumerate(MODELS):
            y=[next(r["accuracy"] for r in groups[field] if r[field]==v and r["model"]==m)*100 for v in values]
            ax.plot(range(len(values)),y,"o-",label=NAMES[m],color=COLORS[i])
        ax.set_xticks(range(len(values)),values);ax.set(ylabel="Accuracy (%)",ylim=(45,100),title="By format" if field=="question_type" else "By difficulty label")
        ax.grid(alpha=.2)
    axes[1].legend(ncol=2,fontsize=8);fig.tight_layout();savefig("format_difficulty",fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    counts=[r["questions"] for r in overlap]
    axes[0].bar(range(5),counts,color=["#ac5151","#df9b35","#76a8bc","#4278b6","#278779"])
    for i,v in enumerate(counts):axes[0].text(i,v+25,str(v),ha="center")
    axes[0].set(xlabel="Number of correct models on the same question",ylabel="Questions",xticks=range(5),title="Agreement and complementarity",ylim=(0,max(counts)*1.12))
    rescue=np.zeros((4,4))
    for i in range(4):
        for j in range(4):rescue[i,j]=np.sum((matrix[:,i]==0)&(matrix[:,j]==1))/np.sum(matrix[:,i]==0)*100
    im=axes[1].imshow(rescue,cmap="Blues",vmin=0,vmax=100)
    for i in range(4):
        for j in range(4):axes[1].text(j,i,f"{rescue[i,j]:.1f}%",ha="center",va="center",color="white" if rescue[i,j]>50 else "#243746")
    axes[1].set_xticks(range(4),[NAMES[m] for m in MODELS]);axes[1].set_yticks(range(4),[NAMES[m] for m in MODELS]);axes[1].set(xlabel="Rescuing model",ylabel="Model that failed",title="Fraction of row-model errors rescued")
    fig.tight_layout();savefig("complementarity",fig)

    fig,ax=plt.subplots(figsize=(8.5,4))
    for i,m in enumerate(MODELS):
        r=next(r for r in cohort_changes if r["model"]==m)
        ax.plot([0,1],[r["old_accuracy"]*100,r["new_accuracy"]*100],"o-",label=NAMES[m],color=COLORS[i])
    ax.set(xticks=[0,1],xticklabels=["Original 1,000","Additional 2,000"],ylabel="Accuracy (%)",ylim=(77,94),title="Cohorts regraded with the same scorer")
    ax.legend(ncol=2);ax.grid(alpha=.2);fig.tight_layout();savefig("cohort_accuracy",fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for mu in THRESHOLDS:
        a=[r for r in traces if r["threshold"]==mu]
        style={.89:"-",.85:"--",.80:":",.825:"-."}[mu]
        axes[0].plot([r["round"] for r in a],[r["cost_regret_usd"] for r in a],style,label=f"threshold={mu*100:g}%")
        axes[1].plot([r["round"] for r in a],[r["quality_regret"] for r in a],style,label=f"threshold={mu*100:g}%")
    axes[0].set(title="Empirical cost pseudo-regret",ylabel="USD")
    axes[1].set(title="Empirical quality pseudo-regret",ylabel="Cumulative probability-gap units")
    for ax in axes:ax.set_xlabel("Decision round t");ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();savefig("ft_ucb_regret",fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for j,m in enumerate(MODELS):
        bottom=np.zeros(4) if j==0 else sum(np.array([r[k] for r in primary]) for k in ["haiku_count","luna_count","gemini_count","sonnet_count"][:j])
        key=["haiku_count","luna_count","gemini_count","sonnet_count"][j]
        axes[0].bar(range(4),[r[key] for r in primary],bottom=bottom,color=COLORS[j],label=NAMES[m])
    axes[0].set(xticks=range(4),xticklabels=[f"{mu*100:g}%" for mu in THRESHOLDS],ylabel="Selections / 3,000",title="FT-UCB primary replay (seed 42)")
    axes[0].legend(ncol=2,fontsize=8)
    for mu in THRESHOLDS:
        a=[r for r in traces if r["threshold"]==mu]
        axes[1].plot([r["round"] for r in a],[r["cumulative_accuracy"]*100 for r in a],label=f"{mu*100:g}%",alpha=.8)
    axes[1].set(xlim=(100,3000),ylim=(70,98),xlabel="Decision round t",ylabel="Cumulative accuracy (%)",title="Cumulative routed accuracy")
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2);fig.tight_layout();savefig("ft_ucb_behavior",fig)

    # A final byte-level check covers the original Word file, all CSVs and journals.
    after = {str(p): sha(p) for p in source_paths}
    assert before == after, "An input changed during analysis"
    validation = {
        "unique_questions": 3000, "observations": 12000, "models": MODELS,
        "questions_per_area": 100, "areas": 30, "duplicate_pairs": 0,
        "overlapping_cohort_ids": 0, "duplicate_exact_prompt_content": 0,
        "merged_rows_equal_source_concatenation": True, "image_hashes_verified": image_count,
        "source_files_unchanged": True, "source_sha256": after,
        "scorer": SCORER, "scorer_sha256": sha(OUT / "scoring.py"),
        "scorer_origin": "dev/bandit:bandit/data.py",
        "independent_ft_ucb_decisions_verified": 12000,
        "model_api_calls": 0, "seed": SEED, "order_replays_per_threshold": REPEATS,
        "raw_missing_cost_count": sum(r["cost_value"] is None for r in rows),
        "missing_cost_resolved_by_explicit_provider_no_charge": sum(r["cost_source"] == "provider_no_charge" for r in rows),
    }
    summary = {
        "validation": validation, "models": summaries, "groups": groups,
        "cohort_comparison": cohort_changes, "paired_comparisons": pairs,
        "overlap": overlap, "oracles": oracle_rows, "static_thresholds": static,
        "ft_ucb_primary": primary, "ft_ucb_sensitivity": sensitivity,
        "collection_billing": billing,
        "unique_correct_questions_by_model": {MODELS[i]: int(np.sum((matrix[:,i]==1)&(number_correct==1))) for i in range(4)},
        "source_question_counts": dict(Counter(cohorts.values())),
        "question_type_counts": dict(Counter(q["question_type"] for q in questions.values())),
        "difficulty_counts": dict(Counter(q["topic_difficulty"] for q in questions.values())),
    }
    dump("summary.json",summary);dump("validation.json",validation)
    print(json.dumps({"models": [{k:r[k] for k in ("model","correct","accuracy","ci95_low","ci95_high","latest_outcome_cost_usd","needs_review")} for r in summaries],
                      "overlap": overlap,"oracles":oracle_rows,"cohort_comparison":cohort_changes,
                      "ft_ucb_primary":primary,"ft_ucb_sensitivity":sensitivity,
                      "images_verified": image_count}, indent=2))


if __name__ == "__main__":
    main()
