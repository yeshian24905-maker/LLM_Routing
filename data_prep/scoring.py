"""Deterministic answer extraction and MMMU-style open-answer normalization.

Reference: https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/eval_utils.py
Numeric answers are rounded to two decimal places, following MMMU. Unlike the
official multiple-choice fallback, an unparseable answer is never guessed.
"""

import json
import math
import re
import string


ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
NUMBER = re.compile(
    r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?(?![\w.])"
)


def final_answer(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty model answer")
    tagged = ANSWER_TAG.findall(text)
    if tagged:
        answer = tagged[-1].strip()
    else:
        answer = text.strip()
        if answer.startswith("```"):
            lines = answer.splitlines()
            answer = "\n".join(lines[1:-1]).strip() if len(lines) > 2 else answer
        try:
            structured = json.loads(answer)
        except json.JSONDecodeError:
            structured = None
        if isinstance(structured, dict) and isinstance(structured.get("answer"), (str, int, float)):
            answer = str(structured["answer"]).strip()
        else:
            marked = re.findall(r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*(.+)", answer, re.I)
            answer = marked[-1].strip() if marked else answer.splitlines()[-1].strip()
    if not answer:
        raise ValueError("Empty final answer")
    return answer


def parse_prediction(text, question_type, options):
    answer = final_answer(text)
    if question_type == "open":
        return answer
    if question_type != "multiple-choice":
        raise ValueError(f"Unsupported question type: {question_type}")
    letters = string.ascii_uppercase[:len(options)]
    cleaned = answer.strip(" \t\r\n().,:;!*$'\"`")
    if len(cleaned) == 1 and cleaned.upper() in letters:
        return cleaned.upper()
    explicit = re.findall(
        rf"(?:\b(?:option|choice|answer)\s*(?:is\s*|:\s*|=\s*)?|\()([{letters}])(?:\)|\b)",
        answer, re.I,
    )
    if explicit:
        candidates = set(letter.upper() for letter in explicit)
        mentioned = set(re.findall(rf"\b([{letters}])\b", answer))
        if len(candidates) == 1 and not mentioned - candidates:
            return explicit[-1].upper()
        raise ValueError("Ambiguous multiple-choice answer")
    matches = [letters[i] for i, option in enumerate(options) if cleaned.casefold() == option.strip().casefold()]
    if len(matches) == 1:
        return matches[0]
    raise ValueError("Cannot extract a unique option letter")


def numeric_value(value):
    try:
        number = float(str(value).strip().replace(",", ""))
    except (ValueError, TypeError, OverflowError):
        return None
    return round(number, 2) if math.isfinite(number) else None


def is_correct(prediction, gold, question_type):
    answers = gold if isinstance(gold, list) else [gold]
    if question_type == "multiple-choice":
        return int(any(prediction == str(answer).strip().upper() for answer in answers))
    normalized = prediction.casefold().strip().rstrip(".")
    numbers = {numeric_value(match.group()) for match in NUMBER.finditer(normalized)}
    whole_number = numeric_value(normalized)
    if whole_number is not None:
        numbers.add(whole_number)
    for answer in answers:
        number = numeric_value(answer)
        if number is not None:
            if number in numbers:
                return 1
        else:
            expected = str(answer).casefold().strip()
            if expected and re.search(r"(?<!\w)" + re.escape(expected) + r"(?!\w)", normalized):
                return 1
    return 0
