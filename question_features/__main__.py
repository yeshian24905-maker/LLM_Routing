"""Build question features and learned accuracy priors: fetch, embed, predict, cluster."""

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def question_ids(results_csv):
    with Path(results_csv).open(encoding="utf-8-sig", newline="") as stream:
        return sorted({row["question_id"] for row in csv.DictReader(stream)})


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def labels(args, questions):
    """Y[question, model] in questions order, models by ascending cost, and their costs."""
    import numpy as np
    from bandit.data import load_csv_replay
    from bandit.ft_evaluate import arm_statistics, resolve_costs
    table, models, _ = load_csv_replay(args.results_csv, args.scores_csv)
    costs = resolve_costs(models, arm_statistics(table, models), None, True)
    models = sorted(models, key=lambda m: (costs[m], m))
    Y = np.array([[table[q["question_id"]][m].correct for m in models] for q in questions], dtype=float)
    return Y, models, np.array([costs[m] for m in models])


def read_priors(path, predictor, questions, models):
    import numpy as np
    rows = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["predictor"] == predictor:
                rows[row["question_id"]] = [float(row[m]) for m in models]
    if not rows:
        raise ValueError(f"No predictor {predictor!r} in {path}")
    return np.array([rows[q["question_id"]] for q in questions])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/mmmu")
    parser.add_argument("--results-csv", type=Path, default=ROOT / "results_3000.csv")
    parser.add_argument("--scores-csv", type=Path, default=ROOT / "bandit/output/ft-ucb/scored_results.csv")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch", help="Download the MMMU questions and images of the results CSV.")
    commands.add_parser("embed", help="Compute text and image embeddings into embeddings.npz.")
    commands.add_parser("predict", help="Cross-fit the accuracy predictors into priors.csv.")
    cluster = commands.add_parser("cluster", help="Group questions by predicted accuracy profile.")
    cluster.add_argument("--predictor", default="text_image")
    cluster.add_argument("--k", type=int, nargs="+", default=[8, 16, 30])
    args = parser.parse_args(argv)
    data = args.data_dir

    if args.command == "fetch":
        from .mmmu import fetch_questions
        count = fetch_questions(question_ids(args.results_csv), data)
        print(f"Wrote {count} questions to {data / 'questions.jsonl'}")
        return

    import numpy as np
    from .mmmu import read_questions
    questions = read_questions(data)
    ids = [q["question_id"] for q in questions]

    if args.command == "embed":
        from .embed import IMAGE_MODEL, TEXT_MODEL, embed_images, embed_text
        text = embed_text(questions)
        image = embed_images(questions, data)
        np.savez(data / "embeddings.npz", ids=np.array(ids), text=text, image=image,
                 text_model=TEXT_MODEL, image_model=IMAGE_MODEL)
        print(f"Wrote text {text.shape} and image {image.shape} embeddings to {data / 'embeddings.npz'}")

    elif args.command == "predict":
        from .evaluate import predictor_report
        from .predict import PREDICTORS, cross_fit, design_matrices, folds
        stored = np.load(data / "embeddings.npz")
        if list(stored["ids"]) != ids:
            raise ValueError("embeddings.npz does not match questions.jsonl; run embed again")
        Y, models, costs = labels(args, questions)
        subjects = [q["subject"] for q in questions]
        X = design_matrices(questions, stored["text"], stored["image"])
        splits = folds(subjects)
        cheap, strong = 0, int(np.argmax(Y.mean(0)))
        rows, reports = [], {}
        for name in PREDICTORS:
            P = cross_fit(name, Y, subjects, splits, X.get(name))
            reports[name] = predictor_report(P, Y, costs, cheap, strong)
            rows += [{"question_id": q, "predictor": name, **{m: round(float(p), 5) for m, p in zip(models, P[i])}}
                     for i, q in enumerate(ids)]
            r = reports[name]
            print(f"{name:12s} brier={r['brier']:.4f} logloss={r['log_loss']:.4f} upgrade_auc={r['upgrade_auc']:.3f} " +
                  " ".join(f"tau{t}: plug {r[f'tau_{t}']['plug_in_accuracy'] or 0:.3f}/${r[f'tau_{t}']['plug_in_cost'] or 0:.2f} "
                           f"front ${r[f'tau_{t}']['frontier_cost'] or 0:.2f}" for t in (0.85, 0.87)), flush=True)
        write_csv(data / "priors.csv", rows)
        (data / "predictor_metrics.json").write_text(json.dumps(
            {"models": models, "costs_usd": costs.tolist(), "folds": len(splits), "reports": reports}, indent=2) + "\n")
        print(f"Wrote {data / 'priors.csv'} and predictor_metrics.json")

    elif args.command == "cluster":
        from .clusters import describe, learned_groups
        Y, models, _ = labels(args, questions)
        P = read_priors(data / "priors.csv", args.predictor, questions, models)
        for k in args.k:
            groups = learned_groups(P, k)
            write_csv(data / f"groups_{args.predictor}_k{k}.csv",
                      [{"question_id": q, "group": f"g{g:02d}"} for q, g in zip(ids, groups)])
            write_csv(data / f"cluster_summary_{args.predictor}_k{k}.csv", describe(groups, questions, Y, P, models))
            print(f"Wrote {k} groups from {args.predictor} predictions")


if __name__ == "__main__":
    main()
