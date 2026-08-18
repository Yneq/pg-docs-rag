from pathlib import Path
import sys

# Keep `python scripts/demo_rag.py` working while importing project modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag import create_rag_service


service = create_rag_service()


def translate_to_english(query):
    return service.inference.generate(
        f"Translate this PostgreSQL question to clear, formal English, keep technical terms: {query}"
    )


def translate_to_chinese(text):
    return service.inference.generate(
        f"請將下列 PostgreSQL 技術回答翻譯成繁體中文（台灣用語），保持技術術語正確，並使用正式、易讀的文字：\n\n{text}"
    )


if __name__ == "__main__":

    while True:

        question = input("\nAsk a PostgreSQL question (Chinese or English, q to quit): ")

        if question.lower() == "q":
            break

        # 中文 → 英文
        english_query = translate_to_english(question)
        print("\nTranslated query:", english_query)

        result = service.query(english_query)
        print("Distances:", [chunk.distance for chunk in result.chunks])

        # 翻譯回中文
        answer_zh = translate_to_chinese(result.answer)

        print("\n====== FINAL ANSWER ======\n")
        print(answer_zh)
