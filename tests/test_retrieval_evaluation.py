import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_retrieval import load_dataset, score_evaluation


class RetrievalEvaluationTests(unittest.TestCase):
    def test_repository_dataset_is_valid(self):
        dataset = load_dataset(Path("evals/retrieval_cases.json"))

        self.assertEqual(dataset["top_k"], 3)
        self.assertGreaterEqual(len(dataset["cases"]), 20)

    def test_scores_hit_rate_mrr_and_guardrail(self):
        dataset = {
            "name": "test",
            "top_k": 3,
            "minimums": {
                "retrieval_hit_rate": 0.5,
                "guardrail_accuracy": 0.5,
            },
            "cases": [
                {
                    "id": "relevant",
                    "question": "q1",
                    "should_answer": True,
                    "expected_sources": ["expected.html"],
                },
                {
                    "id": "irrelevant",
                    "question": "q2",
                    "should_answer": False,
                    "expected_sources": [],
                },
            ],
        }
        observations = [
            {
                "id": "relevant",
                "predicted_answer": True,
                "sources": ["other.html", "expected.html"],
                "distances": [0.2, 0.3],
            },
            {
                "id": "irrelevant",
                "predicted_answer": False,
                "sources": ["noise.html"],
                "distances": [0.9],
            },
        ]

        report = score_evaluation(dataset, observations)

        self.assertEqual(report["metrics"]["retrieval_hit_rate"], 1.0)
        self.assertEqual(report["metrics"]["mean_reciprocal_rank"], 0.5)
        self.assertEqual(report["metrics"]["guardrail_accuracy"], 1.0)
        self.assertTrue(report["passed"])

    def test_rejects_duplicate_case_ids(self):
        invalid = {
            "name": "bad",
            "top_k": 3,
            "minimums": {
                "retrieval_hit_rate": 0.5,
                "guardrail_accuracy": 0.5,
            },
            "cases": [
                {
                    "id": "same",
                    "question": "q1",
                    "should_answer": True,
                    "expected_sources": ["one.html"],
                },
                {
                    "id": "same",
                    "question": "q2",
                    "should_answer": False,
                    "expected_sources": [],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(invalid), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unique"):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
