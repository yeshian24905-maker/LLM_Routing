"""Cross-fitted predictors of each model's probability to answer a question correctly."""

import numpy as np

from .handcrafted import FEATURES, handcrafted_features


PREDICTORS = ("pooled", "subject", "handcrafted", "text", "text_image")
SUBJECT_KAPPA = 20
EMBEDDING_DIMS = 32


def _pca(matrix, dims):
    """Unsupervised projection on all questions; it never sees a label."""
    centred = matrix - matrix.mean(0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[:dims].T


def design_matrices(questions, text_embeddings, image_embeddings):
    """Feature matrix per learned predictor, rows in the order of `questions`."""
    subjects = sorted({q["subject"] for q in questions})
    onehot = np.array([[q["subject"] == s for s in subjects] for q in questions], dtype=float)
    hand = np.array([[handcrafted_features(q)[f] for f in FEATURES] for q in questions], dtype=float)
    text = _pca(text_embeddings, EMBEDDING_DIMS)
    image = _pca(image_embeddings, EMBEDDING_DIMS)
    return {
        "handcrafted": np.hstack([onehot, hand]),
        "text": np.hstack([onehot, hand, text]),
        "text_image": np.hstack([onehot, hand, text, image]),
    }


def folds(subjects, k=5, seed=0):
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(subjects)), subjects))


def cross_fit(name, Y, subjects, splits, X=None):
    """Out-of-fold P[question, model]: each row comes from a fit that never saw that question.

    pooled: training mean per model. subject: training subject mean shrunk toward the
    pooled mean with SUBJECT_KAPPA pseudo-questions. Others: L2 logistic regression
    per model on X, with the penalty chosen by inner 5-fold log-loss.
    """
    P = np.zeros(Y.shape)
    subjects = np.asarray(subjects)
    for train, test in splits:
        pooled = Y[train].mean(0)
        if name == "pooled":
            P[test] = pooled
        elif name == "subject":
            for s in np.unique(subjects[test]):
                rows = train[subjects[train] == s]
                estimate = (Y[rows].sum(0) + SUBJECT_KAPPA * pooled) / (len(rows) + SUBJECT_KAPPA)
                P[test[subjects[test] == s]] = estimate
        else:
            from sklearn.linear_model import LogisticRegressionCV
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            for j in range(Y.shape[1]):
                model = make_pipeline(StandardScaler(), LogisticRegressionCV(
                    Cs=np.logspace(-4, 0, 9), cv=5, scoring="neg_log_loss", max_iter=5000,
                    l1_ratios=(0,), use_legacy_attributes=False))
                model.fit(X[train], Y[train, j])
                P[test, j] = model.predict_proba(X[test])[:, 1]
    return P
