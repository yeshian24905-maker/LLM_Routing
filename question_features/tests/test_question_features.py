"""Checks for handcrafted features, cross-fitting, plug-in routing and learned groups."""

import math
import unittest

import numpy as np

from question_features.clusters import describe, learned_groups
from question_features.evaluate import routing_by_prediction
from question_features.handcrafted import handcrafted_features, question_text
from question_features.mmmu import image_extension, subject_of
from question_features.predict import cross_fit, folds


def question(text, options=(), images=(), kind="multiple-choice", subject="Math"):
    return {"question_id": "q", "subject": subject, "question": text, "options": list(options),
            "images": list(images), "question_type": kind}


class MmmuTests(unittest.TestCase):
    def test_subject_keeps_inner_underscores(self):
        self.assertEqual(subject_of("test_Architecture_and_Engineering_12"), "Architecture_and_Engineering")

    def test_image_extension_from_magic_bytes(self):
        self.assertEqual(image_extension(b"\x89PNG\r\n"), ".png")
        self.assertEqual(image_extension(b"\xff\xd8\xff"), ".jpg")
        self.assertEqual(image_extension(b"GIF89a"), ".img")


class HandcraftedTests(unittest.TestCase):
    def test_counts(self):
        q = question("Find $x^2$ when \\frac{1}{2} <image 1> and 3.5", ["12", "<image 2>"], ["a.png", "b.png"])
        f = handcrafted_features(q)
        self.assertEqual(f["n_images"], 2)
        self.assertEqual(f["n_latex"], 2)
        self.assertEqual(f["n_numbers"], 5)
        self.assertEqual(f["n_options"], 2)
        self.assertEqual(f["is_open"], 0)
        self.assertEqual(f["options_have_images"], 1)
        self.assertIn("B. <image 2>", question_text(q))
        self.assertAlmostEqual(f["log_chars"], math.log1p(len(question_text(q)) - len("<image 1>") - len("<image 2>")))


class CrossFitTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.subjects = ["a"] * 50 + ["b"] * 50
        self.Y = np.column_stack([np.r_[np.ones(50), np.zeros(50)], rng.integers(0, 2, 100)]).astype(float)
        self.splits = folds(self.subjects, k=5)

    def test_every_question_is_predicted_by_a_fit_that_excludes_it(self):
        tested = np.concatenate([test for _, test in self.splits])
        self.assertEqual(sorted(tested), list(range(100)))
        for train, test in self.splits:
            self.assertFalse(set(train) & set(test))

    def test_subject_predictor_separates_subjects_and_shrinks(self):
        P = cross_fit("subject", self.Y, self.subjects, self.splits)
        self.assertTrue((P[:50, 0] > 0.6).all() and (P[:50, 0] < 1).all())
        self.assertTrue((P[50:, 0] < 0.4).all() and (P[50:, 0] > 0).all())

    def test_logistic_predictor_learns_a_feature(self):
        X = np.array([[s == "a"] for s in self.subjects], dtype=float)
        P = cross_fit("handcrafted", self.Y, self.subjects, self.splits, X)
        self.assertGreater(P[:50, 0].mean() - P[50:, 0].mean(), 0.5)


class RoutingTests(unittest.TestCase):
    def test_perfect_prediction_reaches_threshold_at_lowest_cost(self):
        Y = np.array([[1, 1], [0, 1], [1, 1], [0, 1]], dtype=float)
        plug, front = routing_by_prediction(Y.copy(), Y, np.array([1.0, 4.0]), 1.0)
        self.assertEqual(plug, (1.0, 2500.0))
        self.assertEqual(front, (1.0, 2500.0))

    def test_unreachable_threshold_returns_none(self):
        Y = np.array([[0, 0]], dtype=float)
        self.assertEqual(routing_by_prediction(Y, Y, np.array([1.0, 2.0]), 0.5), (None, None))


class ClusterTests(unittest.TestCase):
    def test_profiles_split_into_their_groups(self):
        P = np.array([[0.9, 0.9]] * 5 + [[0.2, 0.9]] * 5)
        groups = learned_groups(P, 2)
        self.assertEqual(len(set(groups[:5])), 1)
        self.assertNotEqual(groups[0], groups[5])
        Y = np.array([[1, 1]] * 5 + [[0, 1]] * 5, dtype=float)
        qs = [question("x", subject="a")] * 5 + [question("y", subject="b")] * 5
        rows = describe(groups, qs, Y, P, ["cheap", "strong"])
        self.assertEqual(sorted(r["size"] for r in rows), [5, 5])
        self.assertEqual({r["acc cheap"] for r in rows}, {1.0, 0.0})


if __name__ == "__main__":
    unittest.main()
