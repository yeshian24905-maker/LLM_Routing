"""Offline regression checks for sampling new questions without changing old data."""

import shutil
import sys
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import pyarrow.parquet as pq

from data_prep.__main__ import main
from data_prep.additional import prepare_additional, validate_additional
from data_prep.common import read_json, read_jsonl, sha256_file, write_json
from data_prep.dataset import download_dataset, question_fingerprint
from data_prep.tests.test_pipeline import TemporaryWorkspace, write_parquet


class AdditionalSamplingTests(TemporaryWorkspace):
    def setUp(self):
        super().setUp()
        hub = self.fake_hub(per_subject=5, bad={"test_Physics_5"})
        for name, rows in hub.files.items():
            for row in rows:
                row["question"] += f" Problem {row['id']}."
            write_parquet(hub.root / name, rows)
        with patch.dict(sys.modules, {"huggingface_hub": hub.module()}), redirect_stdout(StringIO()):
            self.parent = download_dataset(
                self.data_dir, subjects=["Accounting", "Physics"], sample_size=3, seed=7,
            )
        self.snapshot = self.data_dir / ".cache/hub/datasets--MMMU--MMMU/snapshots/fixed-revision"
        for entry in self.parent["source_files"]:
            target = self.snapshot / entry["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(hub.root / entry["file"], target)
        original_results = self.data_dir / "runs/sequential/results.csv"
        original_results.parent.mkdir(parents=True)
        original_results.write_text("original result data", encoding="utf-8")
        self.preserved = {path: path.read_bytes() for path in (
            self.data_dir / "questions.jsonl", self.data_dir / "dataset_manifest.json", original_results,
        )}
        self.output = self.data_dir / "additional"

    def prepare_additional(self, output=None, target=4, seed=11):
        with patch("httpx.Client.request", side_effect=AssertionError("Preparation must be offline")), redirect_stdout(StringIO()):
            return prepare_additional(self.data_dir, output or self.output, target, seed)

    def test_complements_unequal_existing_counts_without_repeating_ids(self):
        result = self.prepare_additional()
        old = list(read_jsonl(self.data_dir / "questions.jsonl"))
        new = list(read_jsonl(self.output / "questions.jsonl"))
        self.assertEqual((result["source_questions"], result["additional_questions"], result["combined_questions"]), (3, 5, 8))
        self.assertTrue({q["question_id"] for q in old}.isdisjoint(q["question_id"] for q in new))
        self.assertEqual(Counter(q["subject"] for q in old + new), {"Accounting": 4, "Physics": 4})
        self.assertNotIn("test_Physics_5", {q["question_id"] for q in new})
        self.assertEqual(read_json(self.output / "dataset_manifest.json")["resolved_revision"], "fixed-revision")
        self.assertEqual(validate_additional(self.data_dir, self.output, 4), result)
        for path, content in self.preserved.items():
            self.assertEqual(path.read_bytes(), content)

    def test_same_seed_is_reproducible_and_existing_output_is_reused(self):
        self.prepare_additional()
        digest = sha256_file(self.output / "questions.jsonl")
        self.prepare_additional()
        self.assertEqual(sha256_file(self.output / "questions.jsonl"), digest)
        other = self.root / "second"
        self.prepare_additional(output=other)
        self.assertEqual(sha256_file(other / "questions.jsonl"), digest)

    def make_duplicate_content(self, duplicate_original):
        entry = next(item for item in self.parent["source_files"] if item["file"].startswith("Accounting/"))
        path = self.snapshot / entry["file"]
        rows = pq.read_table(path).to_pylist()
        old = list(read_jsonl(self.data_dir / "questions.jsonl"))
        old_ids = {q["question_id"] for q in old}
        candidates = [row for row in rows if row["id"] not in old_ids]
        target = candidates[0]
        target["question"] = (next(q["question"] for q in old if q["subject"] == "Accounting")
                              if duplicate_original else candidates[1]["question"])
        write_parquet(path, rows)
        entry["sha256"] = sha256_file(path)
        write_json(self.data_dir / "dataset_manifest.json", self.parent)
        return target["id"]

    def test_different_id_with_original_content_is_replaced(self):
        duplicate_id = self.make_duplicate_content(duplicate_original=True)
        result = self.prepare_additional()
        new = list(read_jsonl(self.output / "questions.jsonl"))
        self.assertEqual(result["additional_questions"], 5)
        self.assertNotIn(duplicate_id, {q["question_id"] for q in new})
        self.assertEqual(result["identical_content_overlap_with_original"], 0)

    def test_duplicate_content_within_new_batch_is_replaced(self):
        self.make_duplicate_content(duplicate_original=False)
        result = self.prepare_additional()
        new = list(read_jsonl(self.output / "questions.jsonl"))
        self.assertEqual(result["additional_questions"], 5)
        self.assertEqual(len({question_fingerprint(q) for q in new}), 5)
        self.assertEqual(result["identical_content_duplicates_in_additional"], 0)

    def test_existing_output_cannot_be_resampled_with_different_settings(self):
        self.prepare_additional()
        with self.assertRaisesRegex(ValueError, "different sampling settings"):
            self.prepare_additional(seed=12)

    def test_missing_cache_never_falls_back_to_network(self):
        (self.snapshot / self.parent["source_files"][0]["file"]).unlink()
        with self.assertRaisesRegex(ValueError, "no download attempted"):
            self.prepare_additional()

    def test_modified_cache_is_rejected(self):
        (self.snapshot / self.parent["source_files"][0]["file"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.prepare_additional()

    def test_source_directory_cannot_be_overwritten(self):
        with self.assertRaisesRegex(ValueError, "original dataset"):
            self.prepare_additional(output=self.data_dir)

    def test_impossible_quota_does_not_change_original_files(self):
        with self.assertRaisesRegex(ValueError, "usable new questions"):
            self.prepare_additional(target=5)
        for path, content in self.preserved.items():
            self.assertEqual(path.read_bytes(), content)

    def test_prepared_image_tampering_is_detected_on_reuse(self):
        self.prepare_additional()
        question = next(read_jsonl(self.output / "questions.jsonl"))
        image = self.output / next(iter(question["images"].values()))
        image.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.prepare_additional()

    def test_prepare_command_and_collection_dry_run_make_no_network_calls(self):
        with patch("httpx.Client.request", side_effect=AssertionError("Commands must stay offline")), redirect_stdout(StringIO()):
            self.assertEqual(main([
                "prepare-additional", "--source-dir", str(self.data_dir),
                "--output-dir", str(self.output), "--target-per-subject", "4", "--seed", "11",
            ]), 0)
            self.assertEqual(main([
                "run", "--data-dir", str(self.output), "--config", str(self.config_path), "--dry-run",
            ]), 0)
        self.assertFalse((self.output / "runs").exists())
