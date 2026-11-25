"""Offline quality of accuracy predictors: calibration, ranking and plug-in routing cost."""

import numpy as np


def routing_by_prediction(P, Y, costs, threshold):
    """Route each question to argmin cost/max_cost - lambda * P.

    Plug-in: the smallest lambda whose predicted accuracy reaches the threshold,
    chosen without labels. Frontier: the cheapest lambda whose realized accuracy
    reaches it, which peeks at labels for this one scalar. Costs per 1000 questions.
    """
    prices = costs / costs.max()
    rows = np.arange(len(P))
    plug, front = None, None
    for lam in np.geomspace(0.05, 500, 600):
        pick = np.argmin(prices[None, :] - lam * P, 1)
        predicted, realized = P[rows, pick].mean(), Y[rows, pick].mean()
        cost = costs[pick].mean() * 1000
        if plug is None and predicted >= threshold:
            plug = (realized, cost)
        if front is None and realized >= threshold:
            front = (realized, cost)
    return plug, front


def predictor_report(P, Y, costs, cheap, strong, thresholds=(0.825, 0.85, 0.87, 0.89)):
    from sklearn.metrics import roc_auc_score
    eps = 1e-6
    clipped = np.clip(P, eps, 1 - eps)
    upgrade = (Y[:, strong] == 1) & (Y[:, cheap] == 0)
    report = {
        "brier": float(((P - Y) ** 2).mean()),
        "log_loss": float(-(Y * np.log(clipped) + (1 - Y) * np.log(1 - clipped)).mean()),
        "auc_per_model": [float(roc_auc_score(Y[:, j], P[:, j])) if len(set(P[:, j])) > 1 else 0.5
                          for j in range(Y.shape[1])],
        "upgrade_auc": float(roc_auc_score(upgrade, P[:, strong] - P[:, cheap]))
                       if len(set(P[:, strong] - P[:, cheap])) > 1 else 0.5,
    }
    for t in thresholds:
        plug, front = routing_by_prediction(P, Y, costs, t)
        report[f"tau_{t}"] = {"plug_in_accuracy": plug and plug[0], "plug_in_cost": plug and plug[1],
                              "frontier_cost": front and front[1]}
    return report
