"""Prepare disjoint MMMU top-up questions from a previously downloaded cache."""

import random
from collections import Counter
from pathlib import Path

from filelock import FileLock

from .common import read_json, sha256_file, utc_now, write_json, write_jsonl
from .dataset import DATASET_ID, DEFAULT_SEED, SUBJECTS, load_questions, normalize_row, question_fingerprint


def validate_additional(source_dir, output_dir, target_per_subject=100):
    """Validate IDs, per-subject totals, and all question/image checksums offline."""
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    old, new = load_questions(source_dir), load_questions(output_dir)
    parent = read_json(source_dir / "dataset_manifest.json")
    manifest = read_json(output_dir / "dataset_manifest.json")
    if manifest.get("parent_dataset", {}).get("questions_sha256") != sha256_file(source_dir / "questions.jsonl"):
        raise ValueError("Additional questions refer to a different parent dataset")
    if manifest.get("resolved_revision") != parent.get("resolved_revision"):
        raise ValueError("Parent and additional questions must use the same MMMU revision")
    old_ids = {q["question_id"] for q in old}
    new_ids = {q["question_id"] for q in new}
    if old_ids & new_ids:
        raise ValueError("Additional questions overlap the original question IDs")
    old_content = {question_fingerprint(q) for q in old}
    new_content = [question_fingerprint(q) for q in new]
    if old_content.intersection(new_content) or len(set(new_content)) != len(new_content):
        raise ValueError("Additional questions repeat exact question content")
    old_counts, new_counts = Counter(q["subject"] for q in old), Counter(q["subject"] for q in new)
    subjects = parent["subjects"]
    if set(new_counts) - set(subjects) or any(
        old_counts[subject] + new_counts[subject] != target_per_subject for subject in subjects
    ):
        raise ValueError("Combined per-subject counts do not match the requested target")
    if manifest.get("question_count") != len(new):
        raise ValueError("Additional manifest question count does not match its question file")
    return {
        "source_questions": len(old), "additional_questions": len(new),
        "combined_questions": len(old) + len(new), "subjects": len(subjects),
        "target_per_subject": target_per_subject, "overlapping_question_ids": 0,
        "identical_content_overlap_with_original": 0, "identical_content_duplicates_in_additional": 0,
        "additional_images_verified": sum(len(q["images"]) for q in new),
        "source_questions_sha256": sha256_file(source_dir / "questions.jsonl"),
        "additional_questions_sha256": sha256_file(output_dir / "questions.jsonl"),
        "per_subject": {
            subject: {"original": old_counts[subject], "additional": new_counts[subject],
                      "combined": old_counts[subject] + new_counts[subject]}
            for subject in sorted(subjects)
        },
    }


def prepare_additional(source_dir, output_dir, target_per_subject=100, seed=DEFAULT_SEED):
    """Top up each parent subject to a target, without network or model requests.

    Source questions and collection results are never modified. Repeating the same
    request validates and reuses the prepared output instead of drawing new questions.
    """
    import pyarrow.parquet as pq

    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source_dir == output_dir or source_dir.is_relative_to(output_dir):
        raise ValueError("Additional output must not replace or contain the original dataset")
    if type(target_per_subject) is not int or target_per_subject <= 0:
        raise ValueError("target_per_subject must be a positive integer")
    original = load_questions(source_dir)
    parent = read_json(source_dir / "dataset_manifest.json")
    parent_hash = sha256_file(source_dir / "questions.jsonl")
    if parent.get("dataset") != DATASET_ID or not parent.get("resolved_revision"):
        raise ValueError("Source manifest must identify a pinned MMMU dataset revision")
    if parent.get("questions_sha256") != parent_hash or parent.get("question_count") != len(original):
        raise ValueError("Source manifest does not match its question file")
    subjects, splits = parent["subjects"], parent["splits"]
    counts = Counter(q["subject"] for q in original)
    if not subjects or len(set(subjects)) != len(subjects) or set(subjects) - set(SUBJECTS) or set(counts) != set(subjects):
        raise ValueError("Source manifest subjects do not match the original questions")
    quota = {subject: target_per_subject - counts[subject] for subject in subjects}
    if any(value < 0 for value in quota.values()):
        raise ValueError("An original subject already exceeds target_per_subject")
    if not sum(quota.values()):
        raise ValueError("The original dataset already meets the per-subject target")
    output_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_dir / "prepare.lock"), timeout=0):
        if (output_dir / "questions.jsonl").exists() or (output_dir / "dataset_manifest.json").exists():
            existing = read_json(output_dir / "dataset_manifest.json")
            if existing.get("sampling", {}).get("seed") != seed or existing.get("sampling", {}).get("target_per_subject") != target_per_subject:
                raise ValueError("Prepared output uses different sampling settings")
            validation = validate_additional(source_dir, output_dir, target_per_subject)
            write_json(output_dir / "validation.json", validation)
            print(f"Reused and verified {validation['additional_questions']} additional questions")
            return validation

        cache_root = source_dir / ".cache" / "hub"
        snapshot = cache_root / "datasets--MMMU--MMMU" / "snapshots" / parent["resolved_revision"]
        local_files = {}
        for entry in parent["source_files"]:
            path = (snapshot / entry["file"]).resolve()
            if not path.is_relative_to(cache_root.resolve()) or not path.is_file():
                raise ValueError(f"Missing or invalid cached parquet: {entry['file']}; no download attempted")
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"Cached parquet checksum mismatch: {entry['file']}")
            local_files[entry["file"]] = path

        selected, skipped = [], []
        seen = {q["question_id"] for q in original}
        seen_content = {question_fingerprint(q) for q in original}
        for subject in subjects:
            if not quota[subject]:
                continue
            candidates = []
            for split in splits:
                names = sorted(name for name in local_files if name.startswith(f"{subject}/{split}-"))
                if not names:
                    raise ValueError(f"No cached source files for {subject}/{split}")
                for name in names:
                    ids = pq.read_table(local_files[name], columns=["id"]).column("id").to_pylist()
                    candidates.extend((name, split, i, qid) for i, qid in enumerate(ids) if qid not in seen)
            random.Random(f"{seed}:{subject}").shuffle(candidates)
            tables = {name: pq.read_table(local_files[name]) for name in {row[0] for row in candidates}}
            taken = 0
            for name, split, position, qid in candidates:
                if taken == quota[subject]:
                    break
                if qid in seen:
                    continue
                try:
                    question = normalize_row(
                        tables[name].slice(position, 1).to_pylist()[0], subject, split, output_dir,
                        excluded_content=seen_content,
                    )
                except (ValueError, SyntaxError) as exc:
                    skipped.append({"question_id": qid, "reason": str(exc)})
                    continue
                seen.add(qid)
                seen_content.add(question_fingerprint(question))
                selected.append(question)
                taken += 1
            if taken != quota[subject]:
                raise ValueError(f"{subject}: only {taken} usable new questions; need {quota[subject]}")
            print(f"{subject}: original {counts[subject]} + additional {taken} = {target_per_subject}", flush=True)
        random.Random(f"{seed}:order").shuffle(selected)
        write_jsonl(output_dir / "questions.jsonl", selected)
        write_json(output_dir / "dataset_manifest.json", {
            "dataset": DATASET_ID, "requested_revision": parent["resolved_revision"],
            "resolved_revision": parent["resolved_revision"], "created_at": utc_now(),
            "splits": splits, "subjects": subjects,
            "parent_dataset": {"path": str(source_dir), "questions_sha256": parent_hash,
                               "question_count": len(original)},
            "sampling": {"method": "per-subject top-up excluding parent IDs, globally shuffled",
                         "sample_size": len(selected), "seed": seed,
                         "target_per_subject": target_per_subject, "quota_per_subject": quota},
            "question_count": len(selected), "question_ids": [q["question_id"] for q in selected],
            "counts": dict(Counter(f"{q['subject']}/{q['split']}" for q in selected)),
            "question_types": dict(Counter(q["question_type"] for q in selected)),
            "source_files": parent["source_files"], "skipped_rows": skipped,
            "questions_sha256": sha256_file(output_dir / "questions.jsonl"),
        })
        validation = validate_additional(source_dir, output_dir, target_per_subject)
        write_json(output_dir / "validation.json", validation)
        print(f"Prepared and verified {len(selected)} additional questions; no network or model API calls")
        return validation
