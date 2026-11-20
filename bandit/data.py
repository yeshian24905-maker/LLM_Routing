"""Validate the complete response table and assign deterministic, auditable scores."""

import ast
import csv
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


REQUIRED_FIELDS = (
    "question_id", "subject", "question_type", "model", "raw_response",
    "correct_answer", "cost", "error",
)
ANSWER_TAG = re.compile(r"<answer\s*>(.*?)</answer\s*>", re.I | re.S)
NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")
SCORER = "tag-first, option-aware deterministic MMMU-style v2 (not the official evaluator)"


@dataclass(frozen=True)
class Observation:
    question_id: str
    model: str
    correct: int
    cost: float | None


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


def choice_letter(text, options=()):
    match = re.fullmatch(r"[\s*`$<\[(]*([A-Z])[\s*`$>\]).,:;!]*", text.strip(), re.I)
    if match:
        return match.group(1).upper()
    # Match the answer against every option, independently of the ground truth.
    matching = [i for i, option in enumerate(options) if normalize(text) == normalize(option)]
    if len(matching) == 1:
        return chr(ord("A") + matching[0])
    # A labeled answer may include its option text, e.g. "C. the third value".
    # Commas are excluded so "B, C" remains an ambiguous multi-option answer.
    match = re.match(r"^[\s*`$]*\(?([A-Z])(?:[.):]\s+|\)\s+)", text, re.I)
    if match:
        return match.group(1).upper()
    matches = list(re.finditer(
        r"\b(?:final\s+answer|answer|option|choice)\s*(?:is\s*)?[:=]?\s*"
        r"[\s*`$<\[(]*([A-Z])\b", text, re.I,
    ))
    return matches[-1].group(1).upper() if matches else None


def extract_answer(response, question_type, options=()):
    if not response.strip():
        return "", "empty_response"
    tags = ANSWER_TAG.findall(response)
    if tags:
        content = tags[-1].strip()
        if question_type == "multiple-choice":
            answer = choice_letter(content, options)
            return (answer or ""), ("answer_tag" if answer else "unparsed_choice")
        return content, "answer_tag" if content else "empty_answer_tag"
    if re.search(r"<answer\b", response, re.I):
        return "", "incomplete_answer_tag"
    if question_type == "multiple-choice":
        answer = choice_letter(response, options)
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


def grade(row, options=()):
    answers = accepted_answers(row["correct_answer"])
    kind = row["question_type"]
    if kind == "multiple-choice" and any(not re.fullmatch("[A-Z]", a.strip()) for a in answers):
        raise ValueError(f"Invalid ground-truth choice: {answers!r}")
    prediction, extraction = extract_answer(row["raw_response"], kind, options)
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


def load_questions(data_dir, expected_prompts=1000):
    """Read the authoritative MMMU prompt list and verify its downloaded manifest."""
    data_dir = Path(data_dir).resolve()
    path = data_dir / "questions.jsonl"
    raw = path.read_bytes()
    manifest = json.loads((data_dir / "dataset_manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("questions_sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError("questions.jsonl checksum does not match dataset_manifest.json")
    questions = {}
    for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        question = json.loads(line)
        fields = ("question_id", "subject", "question_type", "question", "correct_answer")
        if not isinstance(question, dict) or any(
            not isinstance(question.get(field), str) or not question[field].strip() for field in fields
        ):
            raise ValueError(f"{path}: line {number}: incomplete question")
        qid = question["question_id"]
        if qid in questions:
            raise ValueError(f"{path}: duplicate question_id: {qid}")
        if question["question_type"] not in ("multiple-choice", "open"):
            raise ValueError(f"{path}: unknown question_type for {qid}")
        if question["question_type"] == "multiple-choice" and (
            not isinstance(question.get("options"), list) or not question["options"]
            or len(question["options"]) > 26
            or any(not isinstance(option, str) or not option.strip() for option in question["options"])
        ):
            raise ValueError(f"{path}: missing or invalid options for {qid}")
        images = question.get("images", {})
        if not isinstance(images, dict) or any(not isinstance(value, str) for value in images.values()):
            raise ValueError(f"{path}: invalid images for {qid}")
        for relative_path in images.values():
            image_path = (data_dir / relative_path).resolve()
            if not image_path.is_relative_to(data_dir) or not image_path.is_file():
                raise ValueError(f"{qid}: missing or invalid image path: {relative_path}")
        questions[qid] = question
    if len(questions) != expected_prompts or manifest.get("question_count") != len(questions):
        raise ValueError(f"Expected {expected_prompts} MMMU questions; found {len(questions)}")
    return questions


def load_data(path, questions, labels_path=None):
    """Attach saved model outcomes to the MMMU questions, never create extra rounds."""
    labels = {}
    if labels_path is not None:
        for number, row in read_csv(labels_path, ("question_id", "model", "correct")):
            key = row["question_id"], row["model"]
            if key in labels or row["correct"] not in ("0", "1"):
                raise ValueError(f"{labels_path}: record {number}: duplicate pair or correct is not 0/1")
            labels[key] = int(row["correct"])
    table, scored, seen = {}, [], set()
    for number, row in read_csv(path, REQUIRED_FIELDS):
        key = row["question_id"], row["model"]
        if not all(part.strip() for part in key) or not row["subject"].strip():
            raise ValueError(f"{path}: record {number}: missing question_id, model, or subject")
        if key in seen:
            raise ValueError(f"{path}: duplicate (question_id, model): {key}")
        seen.add(key)
        question = questions.get(key[0])
        if question is None:
            raise ValueError(f"{path}: result refers to a question outside questions.jsonl: {key[0]}")
        if any(row[field] != question[field] for field in ("subject", "question_type", "correct_answer")):
            raise ValueError(f"{path}: result metadata does not match MMMU question {key[0]}")
        prediction, extraction, correct = grade(row, question.get("options", ()))
        reviewed = key in labels
        correct = labels.get(key, correct)
        cost = optional_metric(row["cost"], "cost")
        table.setdefault(key[0], {})[key[1]] = Observation(*key, correct, cost)
        scored.append({
            **row, "extracted_answer": prediction, "extraction": extraction,
            "correct": correct, "score_source": "reviewed" if reviewed else "automatic",
            "needs_review": int(not reviewed and (row["question_type"] == "open" or extraction not in (
                "answer_tag", "empty_response", "request_error",
            ))),
        })
    if len(table) != len(questions):
        raise ValueError(f"Expected results for {len(questions)} MMMU questions; found {len(table)}")
    models = sorted({model for observations in table.values() for model in observations})
    if len(models) != 4:
        raise ValueError("FT-UCB requires results from exactly four models")
    incomplete = [qid for qid, observations in table.items() if set(observations) != set(models)]
    if incomplete:
        raise ValueError(f"Missing model results for {len(incomplete)} prompts, e.g. {incomplete[:3]}")
    if labels.keys() - seen:
        raise ValueError(f"Reviewed labels refer to unknown pairs: {list(labels.keys() - seen)[:3]}")
    return {qid: table[qid] for qid in questions}, models, scored
