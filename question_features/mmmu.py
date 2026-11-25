"""Download the MMMU test questions that the results CSV covers, with their images."""

import ast
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


DATASET_ID = "MMMU/MMMU"
REVISION = "876ce5cb130f7f7e290ce4d9984357737d4db5cf"
COLUMNS = ["id", "question", "options", "question_type", "subfield", "topic_difficulty",
           *(f"image_{i}" for i in range(1, 8))]


def subject_of(question_id):
    """test_Architecture_and_Engineering_12 -> Architecture_and_Engineering."""
    return question_id.split("_", 1)[1].rsplit("_", 1)[0]


def image_extension(data):
    if data.startswith(b"\x89PNG"):
        return ".png"
    if data.startswith(b"\xff\xd8"):
        return ".jpg"
    return ".img"


def fetch_questions(question_ids, out_dir):
    """Write out_dir/questions.jsonl and the original image bytes to out_dir/images/.

    Reads the pinned MMMU revision; the Hugging Face cache keeps the parquet files.
    Raises ValueError when a requested id is missing from the dataset.
    """
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    wanted = set(question_ids)
    found = {}
    files = HfApi().list_repo_files(DATASET_ID, repo_type="dataset", revision=REVISION)
    for subject in sorted({subject_of(q) for q in wanted}):
        shards = sorted(f for f in files if f.startswith(f"{subject}/test-") and f.endswith(".parquet"))
        paths = [hf_hub_download(DATASET_ID, f, repo_type="dataset", revision=REVISION) for f in shards]
        batches = (b for p in paths for b in pq.ParquetFile(p).iter_batches(batch_size=64, columns=COLUMNS))
        for batch in batches:
            for row in batch.to_pylist():
                if row["id"] not in wanted:
                    continue
                images = []
                for i in range(1, 8):
                    value = row[f"image_{i}"]
                    if value and value.get("bytes"):
                        name = f"images/{row['id']}_{i}{image_extension(value['bytes'])}"
                        (out_dir / name).write_bytes(value["bytes"])
                        images.append(name)
                options = row["options"]
                found[row["id"]] = {
                    "question_id": row["id"], "subject": subject, "question_type": row["question_type"],
                    "question": row["question"],
                    "options": ast.literal_eval(options) if isinstance(options, str) else list(options or []),
                    "subfield": row["subfield"], "topic_difficulty": row["topic_difficulty"],
                    "images": images,
                }
        print(f"{subject}: {sum(1 for q in found.values() if q['subject'] == subject)} questions", flush=True)
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"{len(missing)} questions not in MMMU {REVISION[:8]}, e.g. {sorted(missing)[:3]}")
    with (out_dir / "questions.jsonl").open("w", encoding="utf-8") as stream:
        for question_id in sorted(found):
            stream.write(json.dumps(found[question_id], ensure_ascii=False) + "\n")
    return len(found)


def read_questions(out_dir):
    with (Path(out_dir) / "questions.jsonl").open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
