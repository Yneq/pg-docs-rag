import unittest

try:
    from fastapi.testclient import TestClient
except ImportError:  # Lets core tests run before API dependencies are installed.
    TestClient = None

from app.inference import GenerationResult
from app.rag import RagResult, RagSettings, RetrievedChunk


class FakeService:
    settings = RagSettings()
    backend_summary = "backend=ollama, model=test"

    def collection_count(self):
        return 4871

    def query(self, question, top_k):
        chunk = RetrievedChunk(
            document="MVCC uses snapshots.",
            distance=12.5,
            source="mvcc-intro.html",
            title="Concurrency Control",
            chunk_index=2,
            lexical_score=8.5,
            lexical_matched_terms=2,
            lexical_query_terms=2,
            lexical_coverage=1.0,
            fusion_score=0.04,
        )
        return RagResult(
            answer="MVCC uses snapshots. [Source 1]",
            grounded=True,
            chunks=[chunk],
            retrieval_seconds=0.1,
            generation_seconds=0.5,
            generation=GenerationResult(
                text="MVCC uses snapshots. [Source 1]",
                prompt_tokens=100,
                generated_tokens=20,
            ),
        )


@unittest.skipIf(TestClient is None, "FastAPI test dependencies are not installed")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.main import app, get_rag_service

        app.dependency_overrides[get_rag_service] = lambda: FakeService()
        cls.app = app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.app.dependency_overrides.clear()

    def test_health_reports_index_count(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["indexed_chunks"], 4871)

    def test_query_returns_sources_and_metrics(self):
        response = self.client.post(
            "/api/query",
            json={"question": "How does MVCC work?", "top_k": 3},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["grounded"])
        self.assertEqual(body["sources"][0]["source"], "mvcc-intro.html")
        self.assertEqual(body["sources"][0]["lexical_coverage"], 1.0)
        self.assertEqual(body["sources"][0]["lexical_matched_terms"], 2)
        self.assertEqual(body["metrics"]["generated_tokens"], 20)
        self.assertEqual(body["metrics"]["tokens_per_second"], 40.0)

    def test_query_validates_input(self):
        response = self.client.post(
            "/api/query", json={"question": "x", "top_k": 99}
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
