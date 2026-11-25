"""Features a router can read from the question itself, before any model call."""

import math
import re


LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
DOLLAR_SPAN = re.compile(r"\$[^$]+\$")
NUMBER = re.compile(r"\d+(?:\.\d+)?")
IMAGE_MARKER = re.compile(r"<image\s*\d+>")

FEATURES = ("n_images", "log_chars", "n_latex", "n_numbers", "n_options", "is_open", "options_have_images")


def question_text(question):
    """Question and lettered options as one string, as the models saw them."""
    options = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(question["options"]))
    return question["question"] + ("\n" + options if options else "")


def handcrafted_features(question):
    text = question_text(question)
    return {
        "n_images": len(question["images"]),
        "log_chars": math.log1p(len(IMAGE_MARKER.sub("", text))),
        "n_latex": len(LATEX_COMMAND.findall(text)) + len(DOLLAR_SPAN.findall(text)),
        "n_numbers": len(NUMBER.findall(IMAGE_MARKER.sub("", text))),
        "n_options": len(question["options"]),
        "is_open": int(question["question_type"] == "open"),
        "options_have_images": int(any(IMAGE_MARKER.search(o) for o in question["options"])),
    }
