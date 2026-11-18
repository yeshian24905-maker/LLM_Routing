"""Validate the complete response table and assign deterministic, auditable scores."""

import ast
import csv
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


REQUIRED_FIELDS = (
    "question_id", "subject", "question_type", "model", "raw_response",
    "correct_answer", "cost", "latency", "error",
)
ANSWER_TAG = re.compile(r"<answer\s*>(.*?)</answer\s*>", re.I | re.S)
NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")
SCORER = "tag-first, deterministic MMMU-style v1 (not the official evaluator)"


@dataclass(frozen=True)
class Observation:
    question_id: str
    model: str
    correct: int
    cost: float | None
    latency: float | None


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).replace("\u2212", "-").casefold().split())


def accepted_answers(value):
    if value.strip().startswith("["):
        try:
            answers = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Invalid accepted-answer list: {value!r}") from exc
        if not isinstance(answers, list) or not answers or any(
            not isinstance(answer, (str, int, float)) or not str(answer).strip()
            for answer in answers
        ):
            raise ValueError(f"Invalid accepted-answer list: {value!r}")
        return [str(answer) for answer in answers]
    if not value.strip():
        raise ValueError("Ground-truth answer is empty")
    return [value]


def choice_letter(text):
    match = re.fullmatch(r"[\s*`$<\[(]*([A-Z])[\s*`$>\]).,:;!]*", text.strip(), re.I)
    if match:
        return match.group(1).upper()
    matches = list(re.finditer(
        r"\b(?:final\s+answer|answer|option|choice)\s*(?:is\s*)?[:=]?\s*"
        r"[\s*`$<\[(]*([A-Z])\b", text, re.I,
    ))
    return matches[-1].group(1).upper() if matches else None


def extract_answer(response, question_type):
    if not response.strip():
        return "", "empty_response"
    tags = ANSWER_TAG.findall(response)
    if tags:
        content = tags[-1].strip()
        if question_type == "multiple-choice":
            answer = choice_letter(content)
            return (answer or ""), ("answer_tag" if answer else "unparsed_choice")
        return content, "answer_tag" if content else "empty_answer_tag"
    if re.search(r"<answer\b", response, re.I):
        return "", "incomplete_answer_tag"
    if question_type == "multiple-choice":
        answer = choice_letter(response)
        return (answer or ""), ("choice_fallback" if answer else "unparsed_choice")
    return response.strip(), "open_fallback"


def grade_open(prediction, answers):
    """MMMU-style numeric rounding and text matching, restricted to the extracted answer."""
    prediction = normalize(prediction)
    numbers = [round(float(token.replace(",", "")), 2) for token in NUMBER.findall(prediction)]
    for answer in answers:
        answer = normalize(answer)
        try:
            number = float(answer.replace(",", ""))
        except ValueError:
            if re.search(r"(?<!\w)" + re.escape(answer) + r"(?!\w)", prediction):
                return True
        else:
            if math.isfinite(number) and round(number, 2) in numbers:
                return True
    return False


def grade(row):
    answers = accepted_answers(row["correct_answer"])
    kind = row["question_type"]
    if kind == "multiple-choice" and any(not re.fullmatch("[A-Z]", a.strip()) for a in answers):
        raise ValueError(f"Invalid ground-truth choice: {answers!r}")
    prediction, extraction = extract_answer(row["raw_response"], kind)
    if row["error"] and not row["error"].startswith("accounting_error:"):
        return prediction, "request_error", 0
    if not prediction:
        return prediction, extraction, 0
    correct = prediction in [a.strip() for a in answers] if kind == "multiple-choice" else grade_open(prediction, answers)
    return prediction, extraction, int(correct)


def read_csv(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or not set(required).issubset(headers):
            raise ValueError(f"{path}: expected unique columns including {', '.join(required)}")
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: record {number} has the wrong number of fields")
            yield number, row


def optional_metric(value, name):
    if not value.strip():
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {value!r}") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative: {value!r}")
    return result


def load_data(path, expected_prompts=1000, labels_path=None):
    labels = {}
    if labels_path is not None:
        for number, row in read_csv(labels_path, ("question_id", "model", "correct")):
            key = row["question_id"], row["model"]
            if key in labels or row["correct"] not in ("0", "1"):
                raise ValueError(f"{labels_path}: record {number}: duplicate pair or correct is not 0/1")
            labels[key] = int(row["correct"])
    table, scored, seen = {}, [], set()
    metadata = {}
    for number, row in read_csv(path, REQUIRED_FIELDS):
        key = row["question_id"], row["model"]
        if not all(part.strip() for part in key) or not row["subject"].strip():
            raise ValueError(f"{path}: record {number}: missing question_id, model, or subject")
        if key in seen:
            raise ValueError(f"{path}: duplicate (question_id, model): {key}")
        seen.add(key)
        if row["question_type"] not in ("multiple-choice", "open"):
            raise ValueError(f"{path}: unknown question_type: {row['question_type']!r}")
        info = (row["subject"], row["question_type"], row["correct_answer"])
        if key[0] in metadata and metadata[key[0]] != info:
            raise ValueError(f"{path}: inconsistent subject/type/answer for {key[0]}")
        metadata[key[0]] = info
        prediction, extraction, correct = grade(row)
        reviewed = key in labels
        correct = labels.get(key, correct)
        cost = optional_metric(row["cost"], "cost")
        latency = optional_metric(row["latency"], "latency")
        table.setdefault(key[0], {})[key[1]] = Observation(*key, correct, cost, latency)
        scored.append({
            **row, "extracted_answer": prediction, "extraction": extraction,
            "correct": correct, "score_source": "reviewed" if reviewed else "automatic",
            "needs_review": int(not reviewed and (row["question_type"] == "open" or extraction not in (
                "answer_tag", "empty_response", "request_error",
            ))),
        })
    if len(table) != expected_prompts:
        raise ValueError(f"Expected {expected_prompts} distinct prompts; found {len(table)}")
    models = sorted({model for observations in table.values() for model in observations})
    if len(models) < 2:
        raise ValueError("Routing needs at least two models")
    incomplete = [qid for qid, observations in table.items() if set(observations) != set(models)]
    if incomplete:
        raise ValueError(f"Missing model results for {len(incomplete)} prompts, e.g. {incomplete[:3]}")
    if labels.keys() - seen:
        raise ValueError(f"Reviewed labels refer to unknown pairs: {list(labels.keys() - seen)[:3]}")
    return table, models, scored
