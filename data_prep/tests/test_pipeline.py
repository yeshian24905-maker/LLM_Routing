import base64
import csv
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from PIL import Image

from data_prep.__main__ import main
from data_prep.collector import collect_dataset, collect_one, fetch_catalog, reported_metrics
from data_prep.common import (
    DEFAULT_CONFIG, FIELDS, append_attempt, latest_results, read_json, read_jsonl,
    recover_journal, sha256_file, write_json, write_jsonl,
)
from data_prep.dataset import build_messages, local_image_bytes, normalize_row, validate_question
from data_prep.scoring import is_correct, parse_prediction


def source_row(identifier="dev_Accounting_1", **changes):
    row = {
        "id": identifier, "question": "Read the table: <image 1>",
        "options": "['10', '20', '30']", "answer": "B",
        "question_type": "multiple-choice", "image_1": Image.new("RGB", (2, 2), "white"),
        "explanation": "SECRET_EXPLANATION",
    }
    row.update(changes)
    return row


def completion(content="<answer>B</answer>", **changes):
    result = {
        "id": "gen-test-1", "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 123, "completion_tokens": 7, "cost": 0.000314},
    }
    result.update(changes)
    return result


class TemporaryWorkspace(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        self.config_path = self.root / "config.json"
        write_json(self.config_path, read_json(DEFAULT_CONFIG))

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, count=1):
        questions = [
            normalize_row(source_row(f"dev_Accounting_{index + 1}"), "Accounting", "dev", self.data_dir)
            for index in range(count)
        ]
        write_jsonl(self.data_dir / "questions.jsonl", questions)
        write_json(self.data_dir / "dataset_manifest.json", {
            "questions_sha256": sha256_file(self.data_dir / "questions.jsonl"),
        })
        return questions

    def client(self, handler):
        return httpx.Client(base_url="https://openrouter.ai/api/v1/", transport=httpx.MockTransport(handler))


class CommandLineTests(TemporaryWorkspace):
    def test_download_catalog_collect_and_export_commands_work_together(self):
        identifiers = [model["id"] for model in read_json(self.config_path)["models"]]
        posts = []

        def load_dataset(dataset_id, subject, *, split, revision, **kwargs):
            self.assertEqual(dataset_id, "MMMU/MMMU")
            self.assertEqual(revision, "fixed-revision")
            yield source_row(f"{split}_{subject}_1")
            self.fail("Download limit should stop before reading the next question")

        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{
                    "id": identifier, "architecture": {"input_modalities": ["image"]},
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                } for identifier in identifiers]})
            posts.append(request)
            return httpx.Response(200, json=completion())

        hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(
            dataset_info=lambda *args, **kwargs: SimpleNamespace(sha="fixed-revision"),
        ))
        factory = lambda *args, **kwargs: self.client(handler)
        run_dir = self.root / "run"
        catalog_path = self.root / "catalog.json"
        with (
            patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=load_dataset), "huggingface_hub": hub}),
            patch.dict(os.environ),
            patch("data_prep.__main__.api_client", side_effect=factory),
            patch("data_prep.collector.api_client", side_effect=factory),
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(main([
                "download", "--splits", "dev", "--subjects", "Accounting", "Physics",
                "--limit-per-subject", "1", "--output-dir", str(self.data_dir),
            ]), 0)
            self.assertEqual(main([
                "check-models", "--config", str(self.config_path), "--output", str(catalog_path),
            ]), 0)
            self.assertEqual(main([
                "run", "--data-dir", str(self.data_dir), "--config", str(self.config_path),
                "--output-dir", str(run_dir),
            ]), 0)
            (run_dir / "results.csv").unlink()
            self.assertEqual(main(["export", "--run-dir", str(run_dir)]), 0)

        self.assertEqual(len(posts), 8)
        self.assertEqual(set(read_json(catalog_path)["models"]), set(identifiers))
        self.assertEqual(read_json(self.data_dir / "dataset_manifest.json")["question_count"], 2)
        rows = list(read_jsonl(run_dir / "results.jsonl"))
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(set(row) == set(FIELDS) and row["correct"] == 1 for row in rows))
        with (run_dir / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 8)


class DatasetAndPromptTests(TemporaryWorkspace):
    def test_multiple_images_in_question_and_options_are_preserved(self):
        row = source_row(
            question="Compare <image 1> and again <image 1>",
            options="['<image 2>', '20', '30']", image_2=Image.new("RGB", (3, 4), "blue"),
        )
        question = normalize_row(row, "Accounting", "dev", self.data_dir)
        messages = build_messages(question, self.data_dir)
        content = messages[1]["content"]
        images = [part for part in content if part["type"] == "image_url"]
        self.assertEqual(len(images), 2)
        self.assertTrue(base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1]).startswith(b"\x89PNG"))
        with Image.open(self.data_dir / question["images"]["2"]) as image:
            self.assertEqual(image.size, (3, 4))
        self.assertNotIn("SECRET_EXPLANATION", json.dumps(messages))

    def test_gold_is_never_put_in_the_prompt(self):
        question = normalize_row(source_row(
            answer="UNIQUE_SECRET_GOLD", options="[]", question_type="open",
        ), "Accounting", "dev", self.data_dir)
        self.assertNotIn("UNIQUE_SECRET_GOLD", json.dumps(build_messages(question, self.data_dir)))

    def test_unreferenced_images_are_still_attached(self):
        question = normalize_row(source_row(
            image_2=Image.new("RGB", (1, 1)),
        ), "Accounting", "dev", self.data_dir)
        content = build_messages(question, self.data_dir)[1]["content"]
        self.assertEqual(sum(part["type"] == "image_url" for part in content), 2)
        self.assertIn("<image 2>", "".join(part.get("text", "") for part in content))

    def test_missing_images_and_labels_are_rejected(self):
        for changes in ({"image_1": None}, {"answer": "?"}, {"answer": "Z"}, {"options": "__import__('os')"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                normalize_row(source_row(**changes), "Accounting", "dev", self.data_dir)

    def test_ids_cannot_escape_output_directory(self):
        with self.assertRaises(ValueError):
            normalize_row(source_row(identifier="../../escape"), "Accounting", "dev", self.data_dir)

    def test_image_tampering_and_path_traversal_are_rejected(self):
        question = self.prepare()[0]
        (self.data_dir / question["images"]["1"]).write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "checksum"):
            local_image_bytes(self.data_dir, question, "1")
        question["images"]["1"] = "../config.json"
        with self.assertRaisesRegex(ValueError, "leaves data directory"):
            local_image_bytes(self.data_dir, question, "1")

    def test_unavailable_open_answer_is_rejected(self):
        question = self.prepare()[0]
        question.update(question_type="open", correct_answer=[])
        with self.assertRaisesRegex(ValueError, "unavailable"):
            validate_question(question)


class ScoringTests(unittest.TestCase):
    def test_lowercase_alternatives_are_rejected(self):
        for response in ("option a or b", "choice b or c", "<answer>option a or b</answer>"):
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_prediction(response, "multiple-choice", ["10", "20", "30"])

    def test_complete_option_text_preserves_punctuation(self):
        for response in ("Venus.", "<answer>Venus.</answer>"):
            with self.subTest(response=response):
                self.assertEqual(parse_prediction(response, "multiple-choice", ["Mercury.", "Venus."]), "B")

    def test_common_multiple_choice_formats(self):
        for response in (
            "<answer>B</answer>", "(B)", "(b)", "b.", "Option b", "The answer is B.",
            '{"answer": "B"}', '```json\n{"answer": "B"}\n```',
            "Reasoning mentions A.\n<answer>B</answer>", "20",
        ):
            with self.subTest(response=response):
                self.assertEqual(parse_prediction(response, "multiple-choice", ["10", "20", "30"]), "B")

    def test_ambiguous_or_empty_answers_are_never_guessed(self):
        for response in ("A or B", "Option A or option B", "Option A or B", "I cannot solve it", "<answer></answer>", "Z"):
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_prediction(response, "multiple-choice", ["10", "20", "30"])

    def test_open_numeric_units_commas_scientific_notation_and_aliases(self):
        for prediction, gold in (
            ("1,234 kg", "1234"), ("3.14159", "3.14"), ("3.14.", "3.14"), ("-2.5e3 m", "-2500"),
            ("DNA", ["deoxyribonucleic acid", "DNA"]), ("x = 42", "42"),
        ):
            with self.subTest(prediction=prediction, gold=gold):
                self.assertEqual(is_correct(prediction, gold, "open"), 1)
        self.assertEqual(is_correct("1842", "42", "open"), 0)
        self.assertEqual(is_correct("sodium", "i", "open"), 0)

    def test_only_final_open_answer_is_scored(self):
        prediction = parse_prediction("42 is an intermediate value.\n<answer>17</answer>", "open", [])
        self.assertEqual(is_correct(prediction, "42", "open"), 0)
        self.assertEqual(is_correct(prediction, "17", "open"), 1)


class ApiObservationTests(TemporaryWorkspace):
    def test_reported_cost_wins_and_zero_cost_is_valid(self):
        for cost in (0, 0.123456):
            payload = completion(usage={"prompt_tokens": 123, "completion_tokens": 7, "cost": cost})
            with self.client(lambda request: self.fail("Accounting lookup was not needed")) as client:
                metrics, sources, _, _, missing = reported_metrics(client, payload)
            self.assertEqual(metrics["cost"], cost)
            self.assertEqual(sources["cost"], "usage.cost")
            self.assertEqual(missing, [])

    def test_generation_fallback_uses_native_usage_and_actual_total_cost(self):
        def handler(request):
            self.assertEqual(request.url.path, "/api/v1/generation")
            self.assertEqual(request.url.params["id"], "gen-test-1")
            return httpx.Response(200, json={"data": {
                "total_cost": 0.0042, "native_tokens_prompt": 120, "native_tokens_completion": 17,
                "tokens_prompt": 999, "tokens_completion": 888,
            }})
        with self.client(handler) as client:
            metrics, sources, _, _, missing = reported_metrics(client, {"id": "gen-test-1"})
        self.assertEqual(metrics, {"input_tokens": 120, "output_tokens": 17, "cost": 0.0042})
        self.assertEqual(sources["cost"], "generation.total_cost")
        self.assertEqual(missing, [])

    def test_unreported_cost_stays_null_without_affecting_answer_grade(self):
        question = self.prepare()[0]
        payload = completion(id=None, usage={"prompt_tokens": 123, "completion_tokens": 7})
        with self.client(lambda request: httpx.Response(200, json=payload)) as client:
            attempt = collect_one(client, question, {"id": "test/model"}, [], 1024)
        row = attempt["result"]
        self.assertIsNone(row["cost"])
        self.assertEqual(row["correct"], 1)
        self.assertIn("accounting_error", row["error"])

    def test_accounting_lag_only_retries_read_only_queries(self):
        count = 0
        def handler(request):
            nonlocal count
            count += 1
            self.assertEqual(request.method, "GET")
            return httpx.Response(404, json={"error": {"message": "not ready"}})
        with self.client(handler) as client, patch("data_prep.collector.time.sleep"):
            metrics, _, _, queries, _ = reported_metrics(client, {"id": "gen-test-1"})
        self.assertEqual(count, 3)
        self.assertEqual(len(queries), 3)
        self.assertIsNone(metrics["cost"])

    def test_unparseable_completion_keeps_raw_output_and_billed_cost(self):
        question = self.prepare()[0]
        raw = "I cannot determine which option is right."
        with self.client(lambda request: httpx.Response(200, json=completion(raw))) as client:
            attempt = collect_one(client, question, {"id": "test/model"}, [], 1024)
        row = attempt["result"]
        self.assertEqual(row["raw_response"], raw)
        self.assertEqual(row["cost"], 0.000314)
        self.assertEqual(row["correct"], 0)
        self.assertIsNone(row["prediction"])
        self.assertIn("parsing_error", row["error"])
        self.assertEqual(attempt["response"]["choices"][0]["message"]["content"], raw)

    def test_truncated_response_is_not_labeled_as_correct(self):
        question = self.prepare()[0]
        payload = completion(choices=[{"message": {"content": "<answer>B</answer>"}, "finish_reason": "length"}])
        with self.client(lambda request: httpx.Response(200, json=payload)) as client:
            row = collect_one(client, question, {"id": "test/model"}, [], 1024)["result"]
        self.assertEqual(row["correct"], 0)
        self.assertIn("finish_reason=length", row["error"])

    def test_timeout_is_saved_as_error_without_fabricating_metrics(self):
        question = self.prepare()[0]
        def handler(request):
            raise httpx.ReadTimeout("timed out", request=request)
        with self.client(handler) as client:
            row = collect_one(client, question, {"id": "test/model"}, [], 1024)["result"]
        self.assertEqual(row["correct"], 0)
        self.assertIn("ReadTimeout", row["error"])
        self.assertTrue(all(row[key] is None for key in ("cost", "input_tokens", "output_tokens")))

    def test_catalog_rejects_missing_models_or_text_only_models(self):
        for entries in ([], [{"id": "test/model", "architecture": {"input_modalities": ["text"]}}]):
            with self.subTest(entries=entries), self.client(lambda request: httpx.Response(200, json={"data": entries})) as client:
                with self.assertRaises(ValueError):
                    fetch_catalog(client, [{"id": "test/model"}])


class ResumeAndExportTests(TemporaryWorkspace):
    def test_four_model_pipeline_resume_retry_and_exact_export_schema(self):
        self.prepare(count=2)
        identifiers = [model["id"] for model in read_json(self.config_path)["models"]]
        calls = []
        failing = {identifiers[1]}
        def handler(request):
            calls.append((request.method, request.url.path))
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{
                    "id": identifier, "architecture": {"input_modalities": ["text", "image"]},
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                } for identifier in identifiers]})
            self.assertEqual(request.url.path, "/api/v1/chat/completions")
            body = json.loads(request.content)
            self.assertTrue(any(part["type"] == "image_url" for part in body["messages"][1]["content"]))
            answer = "<answer></answer>" if body["model"] in failing else "<answer>B</answer>"
            return httpx.Response(200, json=completion(answer))
        run_dir = self.root / "run"
        factory = lambda *args, **kwargs: self.client(handler)
        with patch("data_prep.collector.api_client", side_effect=factory):
            first = collect_dataset(self.data_dir, run_dir, self.config_path)
            self.assertEqual(first["errors"], 2)
            self.assertEqual(first["requested"], 8)
            rows = list(read_jsonl(run_dir / "results.jsonl"))
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(set(row) == set(FIELDS) for row in rows))
            call_count = len(calls)
            resumed = collect_dataset(self.data_dir, run_dir, self.config_path)
            self.assertEqual(resumed["requested"], 0)
            self.assertEqual(len(calls), call_count)
            failing.clear()
            retried = collect_dataset(self.data_dir, run_dir, self.config_path, retry_errors=True)
            self.assertEqual(retried["requested"], 2)
        latest = latest_results(run_dir / "attempts.jsonl")
        self.assertEqual(len(latest), 8)
        self.assertTrue(all(row["correct"] == 1 and row["error"] is None for row in latest.values()))
        self.assertEqual(len(list(read_jsonl(run_dir / "attempts.jsonl"))), 10)
        summary = read_json(run_dir / "summary.json")
        self.assertEqual(summary["attempts"], 10)
        self.assertAlmostEqual(summary["known_cost_usd_all_attempts"], 10 * 0.000314)
        with (run_dir / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        self.assertEqual(len(csv_rows), 8)
        self.assertEqual(tuple(csv_rows[0]), FIELDS)

    def test_settings_change_cannot_mix_billable_experiments(self):
        self.prepare()
        identifiers = [model["id"] for model in read_json(self.config_path)["models"]]
        posts = []
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{
                    "id": identifier, "architecture": {"input_modalities": ["image"]},
                } for identifier in identifiers]})
            posts.append(request)
            return httpx.Response(200, json=completion())
        with patch("data_prep.collector.api_client", side_effect=lambda *args, **kwargs: self.client(handler)):
            collect_dataset(self.data_dir, self.root / "run", self.config_path)
            config = read_json(self.config_path)
            config["max_output_tokens"] += 1
            write_json(self.config_path, config)
            with self.assertRaisesRegex(ValueError, "settings or data changed"):
                collect_dataset(self.data_dir, self.root / "run", self.config_path)
        self.assertEqual(len(posts), 4)

    def test_interrupt_before_first_response_does_not_mask_keyboard_interrupt(self):
        self.prepare()
        identifiers = [model["id"] for model in read_json(self.config_path)["models"]]
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{
                    "id": identifier, "architecture": {"input_modalities": ["image"]},
                } for identifier in identifiers]})
            raise KeyboardInterrupt()
        with patch("data_prep.collector.api_client", side_effect=lambda *args, **kwargs: self.client(handler)):
            with self.assertRaises(KeyboardInterrupt):
                collect_dataset(self.data_dir, self.root / "run", self.config_path)
        self.assertFalse((self.root / "run" / "attempts.jsonl").exists())

    def test_dry_run_has_no_network_or_synthetic_observations(self):
        self.prepare()
        run_dir = self.root / "dry_run"
        with patch("data_prep.collector.api_client", side_effect=AssertionError("Network access in dry run")):
            result = collect_dataset(self.data_dir, run_dir, self.config_path, dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertFalse(run_dir.exists())

    def test_changed_dataset_is_rejected_before_api_access(self):
        self.prepare()
        with (self.data_dir / "questions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("{}\n")
        with patch("data_prep.collector.api_client", side_effect=AssertionError("Should fail locally")):
            with self.assertRaisesRegex(ValueError, "checksum"):
                collect_dataset(self.data_dir, self.root / "run", self.config_path)

    def test_credit_error_stops_after_preserving_the_first_attempt(self):
        self.prepare(count=2)
        identifiers = [model["id"] for model in read_json(self.config_path)["models"]]
        posts = []
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{
                    "id": identifier, "architecture": {"input_modalities": ["image"]},
                } for identifier in identifiers]})
            posts.append(request)
            return httpx.Response(402, json={"error": {"message": "insufficient credits"}})
        with patch("data_prep.collector.api_client", side_effect=lambda *args, **kwargs: self.client(handler)):
            result = collect_dataset(self.data_dir, self.root / "run", self.config_path)
        self.assertEqual(len(posts), 1)
        self.assertEqual(result["errors"], 1)
        self.assertEqual(len(list(read_jsonl(self.root / "run" / "results.jsonl"))), 1)

    def test_incomplete_journal_tail_is_preserved_and_repaired(self):
        journal = self.root / "attempts.jsonl"
        append_attempt(journal, {"result": {"a": 1}})
        with journal.open("ab") as stream:
            stream.write(b'{"result":{"a":')
        recover_journal(journal)
        self.assertTrue(journal.with_name(journal.name + ".partial").exists())
        append_attempt(journal, {"result": {"a": 2}})
        self.assertEqual(len(list(read_jsonl(journal))), 2)

    def test_corrupt_complete_journal_record_is_not_silently_removed(self):
        journal = self.root / "attempts.jsonl"
        journal.write_text("{invalid}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Corrupt journal"):
            recover_journal(journal)


if __name__ == "__main__":
    unittest.main()
