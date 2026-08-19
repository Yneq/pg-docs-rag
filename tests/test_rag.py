import unittest

from app.inference import GenerationResult
from app.rag import (
    REFUSAL_MESSAGE,
    RagService,
    RagSettings,
    expand_postgresql_terms,
)


class FakeCollection:
    def __init__(self, distance=0.25):
        self.distance = distance

    def count(self):
        return 4871

    def query(self, **kwargs):
        return {
            "documents": [["MVCC gives each statement a database snapshot."]],
            "distances": [[self.distance]],
            "metadatas": [[{
                "source": "mvcc-intro.html",
                "title": "Concurrency Control",
                "chunk_index": 2,
            }]],
        }


class FakeInference:
    def __init__(self):
        self.prompt = None

    def generate(self, prompt):
        return self.generate_with_metrics(prompt).text

    def generate_with_metrics(self, prompt):
        self.prompt = prompt
        return GenerationResult(
            text="MVCC uses snapshots. [Source 1]",
            prompt_tokens=60,
            generated_tokens=8,
        )


def make_service(distance=0.25):
    inference = FakeInference()
    service = RagService(
        collection=FakeCollection(distance),
        embed_query=lambda question: [0.1, 0.2],
        inference=inference,
        settings=RagSettings(),
        backend_summary="backend=fake, model=fake",
    )
    return service, inference


class RagServiceTests(unittest.TestCase):
    def test_query_returns_metadata_and_source_grounded_prompt(self):
        service, inference = make_service()

        result = service.query("How does MVCC work?")

        self.assertTrue(result.grounded)
        self.assertEqual(result.chunks[0].source, "mvcc-intro.html")
        self.assertEqual(result.chunks[0].chunk_index, 2)
        self.assertEqual(result.generation.generated_tokens, 8)
        self.assertIn("[Source 1: Concurrency Control]", inference.prompt)
        self.assertIn("using ONLY the provided context", inference.prompt)

    def test_irrelevant_query_refuses_without_generation(self):
        service, inference = make_service(distance=0.61)

        result = service.query("What is the weather?")

        self.assertFalse(result.grounded)
        self.assertEqual(result.answer, REFUSAL_MESSAGE)
        self.assertIsNone(result.generation)
        self.assertIsNone(inference.prompt)

    def test_common_postgresql_acronyms_are_expanded_for_retrieval(self):
        expanded = expand_postgresql_terms("How do MVCC and WAL work?")

        self.assertIn("MVCC (multiversion concurrency control)", expanded)
        self.assertIn("WAL (write-ahead logging)", expanded)

    def test_existing_full_form_is_not_duplicated(self):
        question = "How does MVCC multiversion concurrency control work?"

        self.assertEqual(expand_postgresql_terms(question), question)


if __name__ == "__main__":
    unittest.main()
