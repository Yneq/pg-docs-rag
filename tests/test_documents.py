import unittest

from app.documents import ParsedDocument, chunk_document


class ChunkDocumentTests(unittest.TestCase):
    def test_chunks_include_title_and_overlap(self):
        document = ParsedDocument("MVCC", "A" * 80 + "\n" + "B" * 80)
        chunks = chunk_document(document, chunk_size=100, overlap=20)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(chunk.startswith("Document: MVCC\n\n") for chunk in chunks))
        first_body = chunks[0].split("\n\n", 1)[1]
        second_body = chunks[1].split("\n\n", 1)[1]
        self.assertEqual(first_body[-20:], second_body[:20])

    def test_rejects_invalid_overlap(self):
        document = ParsedDocument("Example", "body")
        with self.assertRaises(ValueError):
            chunk_document(document, chunk_size=100, overlap=100)


if __name__ == "__main__":
    unittest.main()
