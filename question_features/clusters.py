"""Learned routing groups: k-means on the predicted accuracy profile of each question."""

from collections import Counter

import numpy as np

from .handcrafted import FEATURES, handcrafted_features


def learned_groups(P, k, seed=0):
    """Cluster questions by the logits of their predicted per-model accuracy."""
    from sklearn.cluster import KMeans
    logits = np.log(np.clip(P, 1e-4, 1 - 1e-4) / np.clip(1 - P, 1e-4, 1))
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(logits)


def describe(labels, questions, Y, P, models):
    """One row per group: size, actual and predicted accuracy per model, top subjects, mean features."""
    rows = []
    for g in sorted(set(labels)):
        idx = np.flatnonzero(labels == g)
        members = [questions[i] for i in idx]
        hand = [handcrafted_features(q) for q in members]
        row = {"group": f"g{g:02d}", "size": len(idx),
               "top_subjects": "; ".join(f"{s} {c}" for s, c in Counter(q["subject"] for q in members).most_common(3))}
        for j, m in enumerate(models):
            row[f"acc {m}"] = round(float(Y[idx, j].mean()), 3)
            row[f"pred {m}"] = round(float(P[idx, j].mean()), 3)
        for f in FEATURES:
            row[f] = round(float(np.mean([h[f] for h in hand])), 2)
        rows.append(row)
    return rows
