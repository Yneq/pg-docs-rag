"""Evaluate retrieval relevance and guardrail behavior on a fixed question set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "evals/retrieval_cases.json",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON result path.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the dataset without Ollama or Chroma.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("name"), str) or not data["name"].strip():
        raise ValueError("Dataset name must be a non-empty string")
    if not isinstance(data.get("top_k"), int) or data["top_k"] <= 0:
        raise ValueError("Dataset top_k must be a positive integer")

    minimums = data.get("minimums")
    if not isinstance(minimums, dict):
        raise ValueError("Dataset minimums must be an object")
    for key in ("retrieval_hit_rate", "guardrail_accuracy"):
        value = minimums.get(key)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"minimums.{key} must be between 0 and 1")

    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset cases must be a non-empty list")
    seen_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each case must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            raise ValueError("Every case id must be non-empty and unique")
        seen_ids.add(case_id)
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"Case {case_id}: question must be non-empty")
        if not isinstance(case.get("should_answer"), bool):
            raise ValueError(f"Case {case_id}: should_answer must be boolean")
        sources = case.get("expected_sources")
        if not isinstance(sources, list) or not all(
            isinstance(source, str) and source for source in sources
        ):
            raise ValueError(
                f"Case {case_id}: expected_sources must contain strings"
            )
        if case["should_answer"] and not sources:
            raise ValueError(
                f"Case {case_id}: answerable cases need expected_sources"
            )
    return data


def score_evaluation(
    dataset: dict[str, Any], observations: list[dict[str, Any]]
) -> dict[str, Any]:
    cases = dataset["cases"]
    if len(observations) != len(cases):
        raise ValueError("Observation count must match dataset case count")

    retrieval_hits = 0
    reciprocal_rank_total = 0.0
    retrieval_case_count = 0
    guardrail_correct = 0
    results: list[dict[str, Any]] = []

    for case, observation in zip(cases, observations):
        if observation.get("id") != case["id"]:
            raise ValueError("Observations must use dataset order and matching ids")
        predicted_answer = bool(observation["predicted_answer"])
        guardrail_match = predicted_answer == case["should_answer"]
        guardrail_correct += int(guardrail_match)

        sources = observation.get("sources") or []
        source_rank = None
        if case["expected_sources"]:
            retrieval_case_count += 1
            for rank, source in enumerate(sources, start=1):
                if source in case["expected_sources"]:
                    source_rank = rank
                    retrieval_hits += 1
                    reciprocal_rank_total += 1 / rank
                    break

        results.append(
            {
                "id": case["id"],
                "should_answer": case["should_answer"],
                "predicted_answer": predicted_answer,
                "guardrail_correct": guardrail_match,
                "source_rank": source_rank,
                "sources": sources,
                "distances": observation.get("distances") or [],
                "lexical_coverages": observation.get("lexical_coverages") or [],
            }
        )

    metrics = {
        "retrieval_hit_rate": retrieval_hits / retrieval_case_count,
        "mean_reciprocal_rank": reciprocal_rank_total / retrieval_case_count,
        "guardrail_accuracy": guardrail_correct / len(cases),
        "retrieval_hits": retrieval_hits,
        "retrieval_cases": retrieval_case_count,
        "guardrail_correct": guardrail_correct,
        "total_cases": len(cases),
    }
    minimums = dataset["minimums"]
    passed = (
        metrics["retrieval_hit_rate"] >= minimums["retrieval_hit_rate"]
        and metrics["guardrail_accuracy"] >= minimums["guardrail_accuracy"]
    )
    return {
        "dataset": dataset["name"],
        "top_k": dataset["top_k"],
        "minimums": minimums,
        "metrics": metrics,
        "passed": passed,
        "results": results,
    }


def print_report(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    print(f"Dataset: {report['dataset']} ({metrics['total_cases']} cases)")
    print(
        f"Retrieval hit@{report['top_k']}: "
        f"{metrics['retrieval_hit_rate']:.1%} "
        f"({metrics['retrieval_hits']}/{metrics['retrieval_cases']})"
    )
    print(f"MRR@{report['top_k']}: {metrics['mean_reciprocal_rank']:.3f}")
    print(
        f"Guardrail accuracy: {metrics['guardrail_accuracy']:.1%} "
        f"({metrics['guardrail_correct']}/{metrics['total_cases']})"
    )
    print(f"Result: {'PASS' if report['passed'] else 'FAIL'}")
    print("\n| Case | Guardrail | Source rank | Best distance | Lexical coverage |")
    print("|---|---|---:|---:|---:|")
    for result in report["results"]:
        distance = result["distances"][0] if result["distances"] else None
        distance_text = f"{distance:.3f}" if distance is not None else "-"
        coverages = result["lexical_coverages"]
        coverage_text = f"{coverages[0]:.0%}" if coverages else "-"
        source_rank = result["source_rank"] or "-"
        guardrail = "PASS" if result["guardrail_correct"] else "FAIL"
        print(
            f"| {result['id']} | {guardrail} | {source_rank} | "
            f"{distance_text} | {coverage_text} |"
        )


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    dataset = load_dataset(dataset_path)
    if args.validate_only:
        print(
            f"Valid dataset: {dataset['name']} "
            f"({len(dataset['cases'])} cases, top_k={dataset['top_k']})"
        )
        return

    from app.rag import create_rag_service

    service = create_rag_service()
    observations = []
    for case in dataset["cases"]:
        chunks = service.retrieve(case["question"], dataset["top_k"])
        observations.append(
            {
                "id": case["id"],
                "predicted_answer": service.is_relevant(chunks),
                "sources": [chunk.source for chunk in chunks],
                "distances": [chunk.distance for chunk in chunks],
                "lexical_coverages": [
                    chunk.lexical_coverage for chunk in chunks
                ],
            }
        )

    report = score_evaluation(dataset, observations)
    print_report(report)
    if args.output:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved evaluation: {output_path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
