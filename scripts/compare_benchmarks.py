"""Print a compact comparison table for RAG benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="Benchmark JSON files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.results:
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            (
                path.stem,
                result["configuration"],
                result["retrieval_seconds"],
                result["generation_seconds"],
                result["total_seconds"],
            )
        )

    print("| Result | Configuration | Retrieval (s) | Generation (s) | Total (s) |")
    print("|---|---|---:|---:|---:|")
    for label, configuration, retrieval, generation, total in rows:
        print(
            f"| {label} | {configuration} | {retrieval:.3f} | "
            f"{generation:.3f} | {total:.3f} |"
        )


if __name__ == "__main__":
    main()
