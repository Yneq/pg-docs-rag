import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_answers import load_dataset, score_evaluation


class AnswerEvaluationTests(unittest.TestCase):
    def test_repository_dataset_is_valid(self):
        dataset = load_dataset(Path("evals/answer_cases.json"))

        self.assertEqual(dataset["top_k"], 3)
        self.assertGreaterEqual(len(dataset["cases"]), 7)

    def test_scores_concepts_citations_and_pre_generation_guardrail(self):
        dataset = {
            "name": "test",
            "top_k": 2,
            "minimums": {
                "concept_coverage": 0.5,
                "citation_accuracy": 1.0,
                "guardrail_accuracy": 1.0,
            },
            "cases": [
                {
                    "id": "relevant",
                    "question": "What is MVCC?",
                    "should_answer": True,
                    "concept_groups": [["snapshot"], ["version"]],
                },
                {
                    "id": "irrelevant",
                    "question": "What is the weather?",
                    "should_answer": False,
                    "concept_groups": [],
                },
            ],
        }
        observations = [
            {
                "id": "relevant",
                "grounded": True,
                "generation_invoked": True,
                "sources": ["mvcc.html"],
                "answer": "MVCC provides a snapshot. [Source 1]",
            },
            {
                "id": "irrelevant",
                "grounded": False,
                "generation_invoked": False,
                "sources": ["noise.html"],
                "answer": "I could not find relevant information.",
            },
        ]

        report = score_evaluation(dataset, observations)

        self.assertEqual(report["metrics"]["concept_coverage"], 0.5)
        self.assertEqual(report["metrics"]["citation_accuracy"], 1.0)
        self.assertEqual(report["metrics"]["guardrail_accuracy"], 1.0)
        self.assertTrue(report["passed"])

    def test_rejects_invalid_dataset(self):
        invalid = {
            "name": "bad",
            "top_k": 3,
            "minimums": {
                "concept_coverage": 0.7,
                "citation_accuracy": 0.8,
                "guardrail_accuracy": 1.0,
            },
            "cases": [
                {
                    "id": "missing-concepts",
                    "question": "What is MVCC?",
                    "should_answer": True,
                    "concept_groups": [],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(invalid), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "concept groups"):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
