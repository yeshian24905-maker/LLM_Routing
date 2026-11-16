import base64
import csv
import io
import json
import os
import random
import sys
import tempfile
import threading
import unittest
import uuid
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from data_prep import dataset
from data_prep.__main__ import main
from data_prep.collector import (
    RequestPacer, collect_dataset, collect_one, fetch_catalog, post_with_retries, reported_metrics, retry_delay,
)
from data_prep.common import (
    DEFAULT_CONFIG, FIELDS, append_attempt, latest_results, load_config, read_json, read_jsonl,
    recover_journal, request_costs, sha256_file, write_json, write_jsonl,
)
from data_prep.dataset import (
    allocate, build_messages, download_dataset, local_image_bytes, normalize_row, prepare_image,
    validate_question,
)
from data_prep.pilot import cheapest_model, run_pilot, write_report


IMAGE_COLUMNS = [f"image_{index}" for index in range(1, 8)]
PARQUET_SCHEMA = pa.schema(
    [("id", pa.string()), ("question", pa.string()), ("options", pa.string()), ("explanation", pa.string())]
    + [(name, pa.struct([("bytes", pa.binary()), ("path", pa.string())])) for name in IMAGE_COLUMNS]
    + [(name, pa.string()) for name in ("img_type", "answer", "topic_difficulty", "question_type", "subfield")]
)


def encode(image, fmt="PNG", **options):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **options)
    return buffer.getvalue()


def image_value(image=None, fmt="PNG"):
    return {"bytes": encode(image or Image.new("RGB", (2, 2), "white"), fmt), "path": None}


def source_row(identifier="test_Accounting_1", **changes):
    row = {
        "id": identifier, "question": "Read the table: <image 1>",
        "options": "['10', '20', '30']", "answer": "B", "question_type": "multiple-choice",
        "image_1": image_value(), "explanation": "SECRET_EXPLANATION",
        "topic_difficulty": "Easy", "subfield": "Finance",
    }
    row.update(changes)
    return row


def completion(content="<answer>B</answer>", **changes):
    result = {
        "id": "gen-test-1",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 123, "completion_tokens": 70, "cost": 0.000314,
            "completion_tokens_details": {"reasoning_tokens": 60},
        },
    }
    result.update(changes)
    return result


def catalog_entries(identifiers, prices=None):
    return {"data": [{
        "id": identifier, "architecture": {"input_modalities": ["text", "image"]},
        "supported_parameters": ["max_tokens", "reasoning", "include_reasoning"],
        "pricing": {
            "prompt": str((prices or {}).get(identifier, 0.000001)),
            "completion": str((prices or {}).get(identifier, 0.000002)),
        },
    } for identifier in identifiers]}


def write_parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    full = [{name: row.get(name) for name in PARQUET_SCHEMA.names} for row in rows]
    pq.write_table(pa.Table.from_pylist(full, schema=PARQUET_SCHEMA), path)


class FakeHub:
    """Stand-in for huggingface_hub serving local parquet files."""

    def __init__(self, root, files):
        self.root, self.files = Path(root), files

    def module(self):
        info = SimpleNamespace(sha="fixed-revision")
        api = SimpleNamespace(
            dataset_info=lambda *args, **kwargs: info,
            list_repo_files=lambda *args, **kwargs: sorted(self.files),
        )

        def download(repo_id, filename, *, repo_type, revision, **kwargs):
            assert (repo_id, repo_type, revision) == ("MMMU/MMMU", "dataset", "fixed-revision")
            return str(self.root / filename)

        return SimpleNamespace(HfApi=lambda: api, hf_hub_download=download)


class TemporaryWorkspace(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        self.config_path = self.root / "config.json"
        write_json(self.config_path, read_json(DEFAULT_CONFIG))
        self.config = load_config(self.config_path)
        self.identifiers = [model["id"] for model in self.config["models"]]
        sleeper = patch("data_prep.collector.time.sleep")
        self.sleep = sleeper.start()
        self.addCleanup(sleeper.stop)

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, count=1):
        questions = [
            normalize_row(source_row(f"test_Accounting_{index + 1}"), "Accounting", "test", self.data_dir)
            for index in range(count)
        ]
        write_jsonl(self.data_dir / "questions.jsonl", questions)
        write_json(self.data_dir / "dataset_manifest.json", {
            "questions_sha256": sha256_file(self.data_dir / "questions.jsonl"),
            "question_count": count,
        })
        return questions

    def client(self, handler):
        return httpx.Client(base_url="https://openrouter.ai/api/v1/", transport=httpx.MockTransport(handler))

    def fake_hub(self, subjects=("Accounting", "Physics"), per_subject=4, bad=()):
        files = {}
        for subject in subjects:
            rows = [
                source_row(f"test_{subject}_{index}", answer="?" if f"test_{subject}_{index}" in bad else "B")
                for index in range(1, per_subject + 1)
            ]
            name = f"{subject}/test-00000-of-00001.parquet"
            write_parquet(self.root / "hub" / name, rows)
            files[name] = rows
        return FakeHub(self.root / "hub", files)

    def run_api(self, handler):
        return patch("data_prep.collector.api_client", side_effect=lambda *args, **kwargs: self.client(handler))


class ImageTests(unittest.TestCase):
    def test_jpeg_and_png_originals_are_kept_byte_for_byte(self):
        for fmt, extension in (("JPEG", ".jpg"), ("PNG", ".png")):
            with self.subTest(fmt=fmt):
                original = encode(Image.new("RGB", (5, 4), "blue"), fmt)
                data, ext, info = prepare_image(original)
                self.assertEqual((data, ext), (original, extension))
                self.assertFalse(info["reencoded"])

    def test_transparent_pixels_become_white_not_black(self):
        rgba = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        rgba.putpixel((0, 0), (255, 0, 0, 255))
        palette = Image.new("P", (4, 4), 0)
        palette.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
        palette.putpixel((0, 0), 1)
        for source in (encode(rgba), encode(palette, transparency=0)):
            with self.subTest():
                data, ext, info = prepare_image(source)
                self.assertEqual(ext, ".png")
                self.assertTrue(info["flattened_transparency"])
                with Image.open(io.BytesIO(data)) as image:
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.getpixel((3, 3)), (255, 255, 255))
                    self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))

    def test_fully_opaque_rgba_png_is_kept_as_is(self):
        original = encode(Image.new("RGBA", (2, 2), (9, 9, 9, 255)))
        data, _, info = prepare_image(original)
        self.assertEqual(data, original)
        self.assertFalse(info["flattened_transparency"] or info["reencoded"])

    def test_oversized_images_are_scaled_within_provider_limits(self):
        data, _, info = prepare_image(encode(Image.new("RGB", (8000, 4), "white")))
        self.assertTrue(info["resized"])
        with Image.open(io.BytesIO(data)) as image:
            self.assertLessEqual(max(image.size), dataset.MAX_IMAGE_SIDE)
        noisy = Image.frombytes("RGB", (300, 300), random.Random(1).randbytes(300 * 300 * 3))
        with patch.object(dataset, "MAX_IMAGE_BYTES", 60_000):
            data, _, info = prepare_image(encode(noisy))
        self.assertTrue(info["resized"])
        self.assertLessEqual(len(data), 60_000)

    def test_exif_orientation_and_cmyk_jpegs_are_normalized(self):
        exif = Image.Exif()
        exif[0x0112] = 6
        data, ext, info = prepare_image(encode(Image.new("RGB", (6, 2)), "JPEG", exif=exif))
        self.assertEqual(ext, ".jpg")
        self.assertTrue(info["exif_rotated"])
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual(image.size, (2, 6))
        data, _, info = prepare_image(encode(Image.new("CMYK", (2, 2)), "JPEG"))
        self.assertTrue(info["reencoded"])
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual(image.mode, "RGB")


class DatasetAndPromptTests(TemporaryWorkspace):
    def test_multiple_images_keep_their_media_types(self):
        row = source_row(
            question="Compare <image 1> and again <image 1>",
            options="['<image 2>', '20', '30']",
            image_2=image_value(Image.new("RGB", (3, 4), "blue"), "JPEG"),
        )
        question = normalize_row(row, "Accounting", "test", self.data_dir)
        content = build_messages(question, self.data_dir)[1]["content"]
        urls = [part["image_url"]["url"] for part in content if part["type"] == "image_url"]
        self.assertEqual(len(urls), 2)
        self.assertTrue(urls[0].startswith("data:image/png;base64,"))
        self.assertTrue(urls[1].startswith("data:image/jpeg;base64,"))
        self.assertTrue(base64.b64decode(urls[1].split(",", 1)[1]).startswith(b"\xff\xd8"))
        self.assertNotIn("SECRET_EXPLANATION", json.dumps(content))

    def test_gold_is_never_put_in_the_prompt(self):
        question = normalize_row(source_row(
            answer="UNIQUE_SECRET_GOLD", options="[]", question_type="open",
        ), "Accounting", "test", self.data_dir)
        self.assertNotIn("UNIQUE_SECRET_GOLD", json.dumps(build_messages(question, self.data_dir)))

    def test_unreferenced_images_are_still_attached(self):
        question = normalize_row(source_row(image_2=image_value()), "Accounting", "test", self.data_dir)
        content = build_messages(question, self.data_dir)[1]["content"]
        self.assertEqual(sum(part["type"] == "image_url" for part in content), 2)
        self.assertIn("<image 2>", "".join(part.get("text", "") for part in content))

    def test_missing_images_and_labels_are_rejected(self):
        for changes in (
            {"image_1": None}, {"answer": "?"}, {"answer": "Z"}, {"options": "__import__('os')"},
            {"image_1": {"bytes": b"not an image", "path": None}},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                normalize_row(source_row(**changes), "Accounting", "test", self.data_dir)

    def test_ids_cannot_escape_output_directory(self):
        with self.assertRaises(ValueError):
            normalize_row(source_row(identifier="../../escape"), "Accounting", "test", self.data_dir)

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


class SamplingTests(TemporaryWorkspace):
    def test_allocation_is_even_and_respects_small_subjects(self):
        available = {f"S{index:02d}": 600 for index in range(30)}
        available["S00"] = 20
        quota = allocate(1000, available, random.Random(0))
        self.assertEqual(sum(quota.values()), 1000)
        self.assertEqual(quota["S00"], 20)
        self.assertLessEqual(max(quota.values()) - min(v for k, v in quota.items() if k != "S00"), 1)
        self.assertEqual(quota, allocate(1000, available, random.Random(0)))
        with self.assertRaises(ValueError):
            allocate(10, {"A": 3, "B": 3}, random.Random(0))

    def download(self, hub, output, **kwargs):
        with patch.dict(sys.modules, {"huggingface_hub": hub.module()}), patch.dict(os.environ), redirect_stdout(StringIO()):
            return download_dataset(self.root / output, **kwargs)

    def test_seeded_balanced_sample_skips_and_replaces_unusable_rows(self):
        hub = self.fake_hub(subjects=("Accounting", "Physics", "Math"), per_subject=6, bad={"test_Math_1", "test_Math_2"})
        manifest = self.download(hub, "a", subjects=["Accounting", "Physics", "Math"], sample_size=12, seed=7)
        self.assertEqual(manifest["question_count"], 12)
        self.assertEqual(manifest["sampling"]["quota_per_subject"], {"Accounting": 4, "Physics": 4, "Math": 4})
        self.assertEqual(manifest["resolved_revision"], "fixed-revision")
        questions = list(read_jsonl(self.root / "a" / "questions.jsonl"))
        self.assertEqual([q["question_id"] for q in questions], manifest["question_ids"])
        self.assertTrue({"test_Math_1", "test_Math_2"}.isdisjoint(manifest["question_ids"]))
        self.assertEqual(sum(q["subject"] == "Math" for q in questions), 4)
        for question in questions:
            self.assertTrue(all((self.root / "a" / path).exists() for path in question["images"].values()))
        same = self.download(hub, "b", subjects=["Accounting", "Physics", "Math"], sample_size=12, seed=7)
        self.assertEqual(same["question_ids"], manifest["question_ids"])
        other = self.download(hub, "c", subjects=["Accounting", "Physics", "Math"], sample_size=12, seed=8)
        self.assertNotEqual(other["question_ids"], manifest["question_ids"])
        skipped = {item["question_id"] for item in manifest["skipped_rows"]}
        self.assertTrue(skipped <= {"test_Math_1", "test_Math_2"})
        self.assertTrue(all("unavailable" in item["reason"] for item in manifest["skipped_rows"]))

    def test_all_questions_and_existing_output_protection(self):
        hub = self.fake_hub(per_subject=3, bad={"test_Physics_3"})
        manifest = self.download(hub, "all", subjects=["Accounting", "Physics"], sample_size=None)
        self.assertEqual(manifest["question_count"], 5)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.download(hub, "all", subjects=["Accounting", "Physics"], sample_size=None)
        with self.assertRaisesRegex(ValueError, "only 2 usable"):
            self.download(hub, "short", subjects=["Accounting", "Physics"], sample_size=6)


class CollectOneTests(TemporaryWorkspace):
    def collect(self, handler, **config_changes):
        question = self.prepare()[0]
        config = {**self.config, **config_changes}
        with self.client(handler) as client:
            return collect_one(client, question, config["models"][0], [], config)

    def test_response_is_recorded_verbatim_without_parsing(self):
        raw = "I think it is B. 20, but maybe A."
        attempt = self.collect(lambda request: httpx.Response(200, json=completion(raw)))
        row = attempt["result"]
        self.assertEqual(tuple(row), FIELDS)
        self.assertEqual(row["raw_response"], raw)
        self.assertEqual(row["correct_answer"], "B")
        self.assertEqual((row["input_tokens"], row["output_tokens"], row["reasoning_tokens"]), (123, 70, 60))
        self.assertEqual((row["finish_reason"], row["error"], row["cost"]), ("stop", None, 0.000314))

    def test_request_uses_shared_reasoning_and_token_settings(self):
        bodies = []
        def handler(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json=completion())
        self.collect(handler)
        self.assertEqual(bodies[0]["reasoning"], {"effort": "low"})
        self.assertEqual(bodies[0]["max_tokens"], 16384)
        self.assertFalse(bodies[0]["stream"])

    def test_truncated_refused_and_empty_answers_are_data_not_errors(self):
        cases = (
            ({"message": {"content": "partial"}, "finish_reason": "length"}, "partial", "length"),
            ({"message": {"content": None, "refusal": "I can't help"}, "finish_reason": "stop"}, "I can't help", "stop"),
            ({"message": {"content": ""}, "finish_reason": "stop"}, "", "stop"),
        )
        for choice, raw, finish in cases:
            with self.subTest(raw=raw):
                row = self.collect(lambda request: httpx.Response(200, json=completion(choices=[choice])))["result"]
                self.assertEqual((row["raw_response"], row["finish_reason"], row["error"]), (raw, finish, None))

    def test_requests_without_a_response_are_retried(self):
        replies = [
            httpx.Response(429, headers={"Retry-After": "3"}, json={"error": {"message": "slow down"}}),
            "timeout",
            httpx.Response(200, json={"error": {"code": 502, "message": "upstream"}}),
            httpx.Response(200, json=completion()),
        ]
        def handler(request):
            reply = replies.pop(0)
            if reply == "timeout":
                raise httpx.ReadTimeout("timed out", request=request)
            return reply
        attempt = self.collect(handler)
        self.assertIsNone(attempt["result"]["error"])
        self.assertEqual(len(attempt["transport_attempts"]), 4)
        self.assertEqual(self.sleep.call_args_list[0].args[0], 3.0)

    def test_retries_stop_after_max_attempts_and_bad_requests_are_not_retried(self):
        calls = []
        def unavailable(request):
            calls.append(request)
            return httpx.Response(503, json={"error": {"message": "down"}})
        row = self.collect(unavailable)["result"]
        self.assertEqual(len(calls), 5)
        self.assertIn("api_error: HTTP 503", row["error"])
        self.assertTrue(all(row[key] is None for key in ("cost", "input_tokens", "raw_response")))
        calls.clear()
        row = self.collect(lambda request: calls.append(request) or httpx.Response(400, json={"error": "bad"}))["result"]
        self.assertEqual(len(calls), 1)
        self.assertIn("HTTP 400", row["error"])

    def test_network_failure_after_all_attempts_is_saved_without_metrics(self):
        def handler(request):
            raise httpx.ConnectError("offline", request=request)
        row = self.collect(handler, max_attempts=2)["result"]
        self.assertIn("ConnectError", row["error"])
        self.assertIn("after 2 attempts", row["error"])
        self.assertIsNotNone(row["latency"])

    def test_choice_level_provider_error_is_an_api_error(self):
        choice = {"message": {"content": ""}, "finish_reason": "error", "error": {"message": "provider crashed"}}
        row = self.collect(lambda request: httpx.Response(200, json=completion(choices=[choice])))["result"]
        self.assertIn("api_error: choice error", row["error"])


class PacingTests(TemporaryWorkspace):
    def test_pacer_spaces_attempts_and_shares_a_cooldown(self):
        clock = [0.0]

        def wait(seconds):
            clock[0] += seconds
            return False

        stop = SimpleNamespace(is_set=lambda: False, wait=wait)
        with patch("data_prep.collector.time.monotonic", side_effect=lambda: clock[0]):
            pacer = RequestPacer(15)
            self.assertTrue(pacer.wait(stop))
            self.assertEqual(clock[0], 0)
            self.assertTrue(pacer.wait(stop))
            self.assertEqual(clock[0], 4)
            pacer.defer(10)
            self.assertTrue(pacer.wait(stop))
            self.assertEqual(clock[0], 14)

    def test_embedded_reset_header_delays_the_retry(self):
        clock = [0.0]
        calls = []

        def wait(seconds):
            clock[0] += seconds
            return False

        def handler(request):
            calls.append(clock[0])
            if len(calls) == 1:
                return httpx.Response(429, json={"error": {"code": 429, "metadata": {
                    "headers": {"X-RateLimit-Reset": "1800000060000", "X-RateLimit-Limit": "20"},
                }}})
            return httpx.Response(200, json=completion())

        stop = SimpleNamespace(is_set=lambda: False, wait=wait)
        with self.client(handler) as client, \
                patch("data_prep.collector.time.monotonic", side_effect=lambda: clock[0]), \
                patch("data_prep.collector.time.time", side_effect=lambda: 1800000000 + clock[0]):
            response, exception, attempts = post_with_retries(client, {}, 5, stop, RequestPacer(15))
        self.assertEqual(calls, [0, 61])
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(exception)
        self.assertEqual(len(attempts), 2)

    def test_retry_delay_reads_a_reset_from_http_headers(self):
        response = httpx.Response(429, headers={"X-RateLimit-Reset": "1800000060"})
        with patch("data_prep.collector.time.time", return_value=1800000000):
            self.assertEqual(retry_delay(1, response), 61)

    def test_stop_during_backoff_prevents_another_request(self):
        calls = []
        stop = threading.Event()

        def interrupted_wait(seconds):
            stop.set()
            return True

        with self.client(lambda request: calls.append(request) or httpx.Response(429, json={"error": "limited"})) as client, \
                patch.object(stop, "wait", side_effect=interrupted_wait):
            response, _, attempts = post_with_retries(client, {}, 5, stop)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(response.status_code, 429)


class AccountingTests(TemporaryWorkspace):
    def test_reported_cost_wins_and_zero_cost_is_valid(self):
        for cost in (0, 0.123456):
            payload = completion(usage={"prompt_tokens": 123, "completion_tokens": 7, "cost": cost})
            with self.client(lambda request: self.fail("Accounting lookup was not needed")) as client:
                metrics, sources, _, _, missing = reported_metrics(client, payload)
            self.assertEqual(metrics["cost"], cost)
            self.assertIsNone(metrics["reasoning_tokens"])
            self.assertEqual(sources["cost"], "usage.cost")
            self.assertEqual(missing, [])

    def test_generation_fallback_uses_native_usage_and_actual_total_cost(self):
        def handler(request):
            self.assertEqual(request.url.path, "/api/v1/generation")
            return httpx.Response(200, json={"data": {
                "total_cost": 0.0042, "native_tokens_prompt": 120, "native_tokens_completion": 17,
                "native_tokens_reasoning": 9, "tokens_prompt": 999,
            }})
        with self.client(handler) as client:
            metrics, sources, _, _, missing = reported_metrics(client, {"id": "gen-test-1"})
        self.assertEqual(metrics, {"input_tokens": 120, "output_tokens": 17, "reasoning_tokens": 9, "cost": 0.0042})
        self.assertEqual(sources["cost"], "generation.total_cost")
        self.assertEqual(missing, [])

    def test_accounting_lag_only_retries_read_only_queries(self):
        methods = []
        def handler(request):
            methods.append(request.method)
            return httpx.Response(404, json={"error": {"message": "not ready"}})
        with self.client(handler) as client:
            metrics, _, _, queries, missing = reported_metrics(client, {"id": "gen-test-1"})
        self.assertEqual(methods, ["GET"] * 5)
        self.assertEqual(len(queries), 5)
        self.assertEqual(missing, ["input_tokens", "output_tokens", "cost"])

    def test_missing_billing_is_refreshed_later_without_asking_the_model_again(self):
        self.prepare()
        posts, billing_ready = [], [False]
        def handler(request):
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json=catalog_entries(self.identifiers))
            if request.url.path.endswith("/generation"):
                if not billing_ready[0]:
                    return httpx.Response(404, json={"error": "not ready"})
                return httpx.Response(200, json={"data": {"total_cost": 0.5}})
            posts.append(request)
            return httpx.Response(200, json=completion(usage={"prompt_tokens": 1, "completion_tokens": 2}))
        run_dir = self.root / "run"
        with self.run_api(handler), redirect_stdout(StringIO()):
            collect_dataset(self.data_dir, run_dir, self.config_path, model_ids=self.identifiers[:1])
            self.assertIn("accounting_error", next(iter(latest_results(run_dir / "attempts.jsonl").values()))["error"])
            billing_ready[0] = True
            result = collect_dataset(self.data_dir, run_dir, self.config_path, model_ids=self.identifiers[:1])
        self.assertEqual(len(posts), 1)
        self.assertEqual(result["requested"], 0)
        row = next(iter(latest_results(run_dir / "attempts.jsonl").values()))
        self.assertEqual((row["cost"], row["error"]), (0.5, None))
        self.assertEqual(request_costs(run_dir / "attempts.jsonl")["completion_requests"], 1)
        self.assertEqual(read_json(run_dir / "summary.json")["all_requests"]["known_cost_usd"], 0.5)


class CatalogTests(TemporaryWorkspace):
    def test_catalog_rejects_missing_text_only_or_unsupported_parameter_models(self):
        model = {"id": "test/model"}
        cases = (
            [],
            [{"id": "test/model", "architecture": {"input_modalities": ["text"]}}],
            [{"id": "test/model", "architecture": {"input_modalities": ["image"]}, "supported_parameters": ["max_tokens"]}],
        )
        for entries in cases:
            with self.subTest(entries=entries), self.client(lambda request: httpx.Response(200, json={"data": entries})) as client:
                with self.assertRaises(ValueError):
                    fetch_catalog(client, [model], self.config)
        with self.client(lambda request: httpx.Response(200, json=catalog_entries(["test/model"]))) as client:
            self.assertIn("test/model", fetch_catalog(client, [model], self.config)["models"])


class RunTests(TemporaryWorkspace):
    def test_retry_errors_preserves_a_recorded_partial_answer(self):
        self.prepare()
        calls = []

        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json=catalog_entries(self.identifiers))
            calls.append(request)
            choice = {"message": {"content": "B. 20"}, "finish_reason": "error", "error": {"code": 502}}
            return httpx.Response(200, json=completion(choices=[choice]))

        run_dir = self.root / "run"
        with self.run_api(handler), redirect_stdout(StringIO()):
            collect_dataset(self.data_dir, run_dir, self.config_path, model_ids=self.identifiers[:1])
            result = collect_dataset(self.data_dir, run_dir, self.config_path,
                                     model_ids=self.identifiers[:1], retry_errors=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["requested"], 0)
        self.assertEqual(next(iter(latest_results(run_dir / "attempts.jsonl").values()))["raw_response"], "B. 20")

    def test_four_models_run_in_parallel_resume_and_retry_only_failed_requests(self):
        self.prepare(count=3)
        failing, empty = self.identifiers[1], self.identifiers[2]
        broken = [True]
        lock, active, peak, posts = threading.Lock(), [0], [0], []
        def handler(request):
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json=catalog_entries(self.identifiers))
            body = json.loads(request.content)
            self.assertTrue(any(part["type"] == "image_url" for part in body["messages"][1]["content"]))
            with lock:
                posts.append(body["model"])
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            threading.Event().wait(0.05)  # time.sleep is patched for the retry logic
            with lock:
                active[0] -= 1
            if body["model"] == failing and broken[0]:
                return httpx.Response(400, json={"error": {"message": "bad request"}})
            content = "" if body["model"] == empty else "<answer>B</answer>"
            return httpx.Response(200, json=completion(content, id=f"gen-{uuid.uuid4()}"))
        run_dir = self.root / "run"
        with self.run_api(handler), redirect_stdout(StringIO()):
            first = collect_dataset(self.data_dir, run_dir, self.config_path, workers_per_model=2)
            self.assertEqual((first["requested"], first["errors"]), (12, 3))
            self.assertGreater(peak[0], 1)
            rows = list(read_jsonl(run_dir / "results.jsonl"))
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(tuple(row) == FIELDS for row in rows))
            count = len(posts)
            self.assertEqual(collect_dataset(self.data_dir, run_dir, self.config_path)["requested"], 0)
            self.assertEqual(len(posts), count)
            broken[0] = False
            retried = collect_dataset(self.data_dir, run_dir, self.config_path, retry_errors=True)
        self.assertEqual(retried["requested"], 3)
        self.assertEqual(posts[count:], [failing] * 3)
        latest = latest_results(run_dir / "attempts.jsonl")
        self.assertTrue(all(row["error"] is None for row in latest.values()))
        self.assertEqual(sum(row["raw_response"] == "" for row in latest.values()), 3)
        summary = read_json(run_dir / "summary.json")
        self.assertEqual(summary["all_requests"]["completion_requests"], 12)
        self.assertEqual(summary["all_requests"]["failed_requests"], 3)
        self.assertEqual(summary["models"][empty]["empty_responses"], 3)
        with (run_dir / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(tuple(next(csv.reader(stream))), FIELDS)

    def test_settings_change_cannot_mix_experiments(self):
        self.prepare()
        handler = lambda request: httpx.Response(
            200, json=catalog_entries(self.identifiers) if request.method == "GET" else completion())
        with self.run_api(handler), redirect_stdout(StringIO()):
            collect_dataset(self.data_dir, self.root / "run", self.config_path)
            config = read_json(self.config_path)
            config["request_parameters"]["reasoning"]["effort"] = "high"
            write_json(self.config_path, config)
            with self.assertRaisesRegex(ValueError, "settings or data changed"):
                collect_dataset(self.data_dir, self.root / "run", self.config_path)

    def test_credit_error_stops_all_workers(self):
        self.prepare(count=5)
        posts = []
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json=catalog_entries(self.identifiers))
            posts.append(request)
            return httpx.Response(402, json={"error": {"message": "insufficient credits"}})
        with self.run_api(handler), redirect_stdout(StringIO()):
            result = collect_dataset(self.data_dir, self.root / "run", self.config_path, workers_per_model=1)
        self.assertLessEqual(len(posts), len(self.identifiers))
        self.assertEqual(result["errors"], len(posts))
        self.assertEqual(result["not_sent"], 20 - len(posts))
        self.assertEqual(len(list(read_jsonl(self.root / "run" / "attempts.jsonl"))), len(posts))

    def test_interrupt_is_not_masked(self):
        self.prepare()
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json=catalog_entries(self.identifiers))
            raise KeyboardInterrupt()
        with self.run_api(handler), redirect_stdout(StringIO()), self.assertRaises(KeyboardInterrupt):
            collect_dataset(self.data_dir, self.root / "run", self.config_path)
        self.assertFalse((self.root / "run" / "attempts.jsonl").exists())

    def test_dry_run_has_no_network_or_synthetic_observations(self):
        self.prepare()
        run_dir = self.root / "dry_run"
        with patch("data_prep.collector.api_client", side_effect=AssertionError("Network access in dry run")), redirect_stdout(StringIO()):
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

    def test_incomplete_journal_tail_is_preserved_and_repaired(self):
        journal = self.root / "attempts.jsonl"
        append_attempt(journal, {"result": {"a": 1}})
        with journal.open("ab") as stream:
            stream.write(b'{"result":{"a":')
        with redirect_stdout(StringIO()):
            recover_journal(journal)
        self.assertTrue(journal.with_name(journal.name + ".partial").exists())
        append_attempt(journal, {"result": {"a": 2}})
        self.assertEqual(len(list(read_jsonl(journal))), 2)

    def test_corrupt_complete_journal_record_is_not_silently_removed(self):
        journal = self.root / "attempts.jsonl"
        journal.write_text("{invalid}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Corrupt journal"):
            recover_journal(journal)


class PilotTests(TemporaryWorkspace):
    def test_pilot_command_retries_only_failed_pairs_with_pacing(self):
        self.prepare(count=2)
        self.config["max_attempts"] = 1
        write_json(self.config_path, self.config)
        calls = []

        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json=catalog_entries(self.identifiers))
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(429, json={"error": {"code": 429, "message": "rate limited"}})
            return httpx.Response(200, json=completion(id=f"gen-{len(calls)}"))

        run_dir = self.root / "run"
        factory = lambda *args, **kwargs: self.client(handler)
        with self.run_api(handler), patch("data_prep.pilot.api_client", side_effect=factory), redirect_stdout(StringIO()):
            summary, _ = run_pilot(self.data_dir, self.config_path, questions=2,
                                   model_id=self.identifiers[0], output_dir=run_dir, workers_per_model=1)
            self.assertEqual(summary["errors"], 1)
            with patch("data_prep.collector.RequestPacer.wait", return_value=True) as paced:
                result = main([
                    "pilot", "--data-dir", str(self.data_dir), "--config", str(self.config_path),
                    "--model", self.identifiers[0], "--n", "2", "--output-dir", str(run_dir),
                    "--retry-errors", "--requests-per-minute", "15", "--workers-per-model", "1",
                ])
            self.assertEqual(paced.call_count, 1)
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 3)
        rows = list(latest_results(run_dir / "attempts.jsonl").values())
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["error"] is None for row in rows))
        self.assertEqual(len(list(read_jsonl(run_dir / "attempts.jsonl"))), 3)

    def test_report_can_be_rebuilt_offline_with_questions_options_and_images(self):
        questions = self.prepare(count=2)
        posts = []

        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json=catalog_entries(self.identifiers))
            posts.append(request)
            return httpx.Response(200, json=completion())

        run_dir = self.root / "review reports"
        with self.run_api(handler), redirect_stdout(StringIO()):
            collect_dataset(self.data_dir, run_dir, self.config_path, model_ids=self.identifiers[:1])
        preserved = {name: (run_dir / name).read_bytes() for name in (
            "attempts.jsonl", "results.jsonl", "results.csv", "summary.json", "run_manifest.json",
        )}
        catalog = {"models": {entry["id"]: entry for entry in catalog_entries(self.identifiers)["data"]}}
        with patch("httpx.Client.request", side_effect=AssertionError("Report generation must remain offline")):
            report, stats = write_report(run_dir, self.identifiers[0], catalog, 2, self.data_dir)
        text = report.read_text(encoding="utf-8")
        self.assertEqual(stats["rows"], 2)
        self.assertEqual(text.count("#### Question\n"), 2)
        self.assertEqual(text.count("#### Recorded response\n"), 2)
        self.assertEqual(text.count("![Image 1]"), 2)
        self.assertIn("A. 10\nB. 20\nC. 30", text)
        self.assertIn("<answer>B</answer>", text)
        self.assertNotIn("SECRET_EXPLANATION", text)
        for question in questions:
            self.assertIn(question["question"], text)
            self.assertIn(f"../data/{question['images']['1']}", text)
        self.assertEqual(len(posts), 2)
        for name, content in preserved.items():
            self.assertEqual((run_dir / name).read_bytes(), content)

    def test_pilot_uses_cheapest_model_and_writes_a_review_report(self):
        self.prepare(count=3)
        prices = {self.identifiers[0]: 0.000005, self.identifiers[2]: 0.0000001}
        posts = []
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json=catalog_entries(self.identifiers, prices))
            posts.append(json.loads(request.content)["model"])
            return httpx.Response(200, json=completion("Thinking...\n```\n<answer>C</answer>"))
        factory = lambda *args, **kwargs: self.client(handler)
        with patch("data_prep.collector.api_client", side_effect=factory), \
                patch("data_prep.pilot.api_client", side_effect=factory), redirect_stdout(StringIO()):
            summary, report = run_pilot(self.data_dir, self.config_path, questions=2)
        self.assertEqual(posts, [self.identifiers[2]] * 2)
        self.assertEqual(summary["requested"], 2)
        self.assertIn("pilot_google-gemini-3-8-flash_2", str(report))
        text = report.read_text(encoding="utf-8")
        self.assertIn("````text\nThinking...\n```\n<answer>C</answer>\n````", text)
        self.assertIn("2 / 2", text)
        self.assertIn("../../images/test_Accounting_1_1.png", text)
        self.assertNotIn("test_Accounting_3", text)
        for identifier in self.identifiers:
            self.assertIn(identifier, text)
        self.assertNotIn("accuracy", text.lower())

    def test_cheapest_model_requires_prices(self):
        with self.assertRaises(ValueError):
            cheapest_model({"models": {"a": {"pricing": {}}}})


class CommandLineTests(TemporaryWorkspace):
    def test_download_check_run_export_commands_work_together(self):
        hub = self.fake_hub(per_subject=2)
        posts = []
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json=catalog_entries(self.identifiers))
            posts.append(request)
            return httpx.Response(200, json=completion())
        factory = lambda *args, **kwargs: self.client(handler)
        run_dir = self.root / "run"
        catalog_path = self.root / "catalog.json"
        data_dir = self.root / "downloaded"
        with (
            patch.dict(sys.modules, {"huggingface_hub": hub.module()}),
            patch.dict(os.environ),
            patch("data_prep.__main__.api_client", side_effect=factory),
            patch("data_prep.collector.api_client", side_effect=factory),
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(main([
                "download", "--subjects", "Accounting", "Physics", "--sample-size", "2",
                "--output-dir", str(data_dir),
            ]), 0)
            self.assertEqual(main(["check-models", "--config", str(self.config_path), "--output", str(catalog_path)]), 0)
            self.assertEqual(main(["run", "--data-dir", str(data_dir), "--config", str(self.config_path), "--dry-run"]), 0)
            self.assertEqual(main([
                "run", "--data-dir", str(data_dir), "--config", str(self.config_path), "--output-dir", str(run_dir),
            ]), 0)
            (run_dir / "results.csv").unlink()
            self.assertEqual(main(["export", "--run-dir", str(run_dir)]), 0)
        self.assertEqual(len(posts), 8)
        self.assertEqual(set(read_json(catalog_path)["models"]), set(self.identifiers))
        manifest = read_json(data_dir / "dataset_manifest.json")
        self.assertEqual(manifest["sampling"]["quota_per_subject"], {"Accounting": 1, "Physics": 1})
        with (run_dir / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 8)


if __name__ == "__main__":
    unittest.main()
