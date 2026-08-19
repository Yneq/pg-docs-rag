import unittest

from app.retrieval import Bm25Index, LexicalChunk, tokenize


def chunk(chunk_id, document):
    return LexicalChunk(
        chunk_id=chunk_id,
        document=document,
        metadata={"source": f"{chunk_id}.html"},
        embedding=[0.0, 0.0],
    )


class Bm25IndexTests(unittest.TestCase):
    def test_tokenizer_removes_common_question_words(self):
        self.assertEqual(
            tokenize("How does PostgreSQL MVCC work?"),
            ["mvcc"],
        )

    def test_exact_technical_terms_rank_relevant_document_first(self):
        index = Bm25Index(
            [
                chunk("vacuum", "VACUUM reclaims dead tuples from a table."),
                chunk("wal", "Write-ahead logging records data changes."),
                chunk("index", "B-tree indexes support equality queries."),
            ]
        )

        results = index.search("How does VACUUM reclaim dead tuples?", 2)

        self.assertEqual(results[0].chunk.chunk_id, "vacuum")
        self.assertGreater(results[0].score, 0)
        self.assertEqual(results[0].matched_terms, 3)
        self.assertEqual(results[0].query_terms, 4)
        self.assertEqual(results[0].coverage, 0.75)

    def test_empty_query_returns_no_results(self):
        index = Bm25Index([chunk("one", "PostgreSQL documentation")])

        self.assertEqual(index.search("How does it work?", 3), [])


if __name__ == "__main__":
    unittest.main()
