"""Full-information optimum: the cheapest randomized routing that meets the accuracy threshold."""

from scipy.optimize import linprog


def oracle_routing(table, domains, models, costs, threshold, per_domain=True):
    """Solve the routing LP from the saved outcomes of every model.

    Each group (one domain, or all questions when per_domain is false) gets a
    probability mix over the models. The LP minimizes the expected cost per
    question subject to expected accuracy >= threshold. Returns cost and
    accuracy per question, or cost None when no mix reaches the threshold.
    """
    group = (lambda q: domains[q]) if per_domain else (lambda q: "all")
    groups = sorted({group(q) for q in table})
    size = {g: 0 for g in groups}
    hits = {(g, m): 0 for g in groups for m in models}
    for question_id, outcomes in table.items():
        size[group(question_id)] += 1
        for model in models:
            hits[group(question_id), model] += outcomes[model].correct
    total = len(table)
    keys = [(g, m) for g in groups for m in models]
    weight = [size[g] / total for g, _ in keys]
    accuracy = [hits[g, m] / size[g] for g, m in keys]
    best = sum(size[g] / total * max(hits[g, m] / size[g] for m in models) for g in groups)
    if best < threshold:
        return {"feasible": False, "cost_usd": None, "accuracy": best}
    solution = linprog(
        c=[w * costs[m] for w, (_, m) in zip(weight, keys)],
        A_ub=[[-w * a for w, a in zip(weight, accuracy)]], b_ub=[-threshold],
        A_eq=[[float(k[0] == g) for k in keys] for g in groups], b_eq=[1.0] * len(groups),
        bounds=(0, 1), method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"Oracle LP failed: {solution.message}")
    reached = sum(w * a * x for w, a, x in zip(weight, accuracy, solution.x))
    return {"feasible": True, "cost_usd": solution.fun, "accuracy": reached}
