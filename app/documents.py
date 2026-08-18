"""Document parsing and deterministic chunking helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    text: str


def parse_document(path: Path) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() not in {".html", ".htm"}:
        return ParsedDocument(title=path.stem, text=_normalize_text(raw))

    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError(
            "HTML parsing requires BeautifulSoup. Install project dependencies "
            "with `pip install -r requirements.txt`."
        ) from exc

    soup = BeautifulSoup(raw, "html.parser")
    for selector in (
        "script",
        "style",
        ".navheader",
        ".navfooter",
        ".indexterm",
        ".id_link",
    ):
        for element in soup.select(selector):
            element.decompose()

    title_element = soup.find("title") or soup.find(["h1", "h2"])
    title = title_element.get_text(" ", strip=True) if title_element else path.stem
    content = soup.body or soup
    return ParsedDocument(
        title=_normalize_text(title),
        text=_normalize_text(content.get_text("\n", strip=True)),
    )


def _normalize_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        normalized = re.sub(r"[ \t\f\v]+", " ", line).strip()
        if normalized:
            lines.append(normalized)
    return "\n".join(lines)


def chunk_document(
    document: ParsedDocument,
    chunk_size: int = 1800,
    overlap: int = 200,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    text = document.text
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            search_from = start + int(chunk_size * 0.6)
            newline_boundary = text.rfind("\n", search_from, end)
            sentence_boundary = text.rfind(". ", search_from, end)
            if newline_boundary >= sentence_boundary and newline_boundary > start:
                end = newline_boundary
            elif sentence_boundary > start:
                end = sentence_boundary + 1

        body = text[start:end].strip()
        if body:
            chunks.append(f"Document: {document.title}\n\n{body}")
        if end >= len(text):
            break
        start = end - overlap

    return chunks
