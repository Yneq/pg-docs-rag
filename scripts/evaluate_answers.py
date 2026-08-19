"""Evaluate end-to-end answer concepts, citations, and guardrail behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "evals/answer_cases.json",
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
    for key in ("concept_coverage", "citation_accuracy", "guardrail_accuracy"):
        value = minimums.get(key)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"minimums.{key} must be between 0 and 1")

    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset cases must be a non-empty list")
    seen_ids: set[str] = set()
    for case in cases:
        case_id = case.get("id") if isinstance(case, dict) else None
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            raise ValueError("Every case id must be non-empty and unique")
        seen_ids.add(case_id)
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"Case {case_id}: question must be non-empty")
        if not isinstance(case.get("should_answer"), bool):
            raise ValueError(f"Case {case_id}: should_answer must be boolean")
        groups = case.get("concept_groups")
        valid_groups = isinstance(groups, list) and all(
            isinstance(group, list)
            and group
            and all(isinstance(term, str) and term for term in group)
            for group in groups
        )
        if not valid_groups and groups != []:
            raise ValueError(
                f"Case {case_id}: concept_groups must contain non-empty string lists"
            )
        if case["should_answer"] and not groups:
            raise ValueError(f"Case {case_id}: answerable cases need concept groups")
        if not case["should_answer"] and groups:
            raise ValueError(f"Case {case_id}: refused cases cannot need concepts")
    if not any(case["should_answer"] for case in cases):
        raise ValueError("Dataset must contain at least one answerable case")
    return data


def _concept_matches(answer: str, groups: list[list[str]]) -> list[bool]:
    normalized = answer.casefold()
    return [
        any(term.casefold() in normalized for term in alternatives)
        for alternatives in groups
    ]


def _valid_citations(answer: str, source_count: int) -> tuple[list[int], bool]:
    citations = [int(value) for value in CITATION_PATTERN.findall(answer)]
    valid = bool(citations) and all(1 <= value <= source_count for value in citations)
    return citations, valid


def score_evaluation(
    dataset: dict[str, Any], observations: list[dict[str, Any]]
) -> dict[str, Any]:
    cases = dataset["cases"]
    if len(observations) != len(cases):
        raise ValueError("Observation count must match dataset case count")

    results = []
    concept_matches = 0
    concept_total = 0
    citation_passes = 0
    answerable_cases = 0
    guardrail_passes = 0

    for case, observation in zip(cases, observations):
        if observation.get("id") != case["id"]:
            raise ValueError("Observations must use dataset order and matching ids")
        grounded = bool(observation["grounded"])
        generation_invoked = bool(observation["generation_invoked"])
        guardrail_correct = (
            grounded and generation_invoked
            if case["should_answer"]
            else not grounded and not generation_invoked
        )
        guardrail_passes += int(guardrail_correct)

        answer = str(observation.get("answer") or "")
        groups = case["concept_groups"]
        matches = _concept_matches(answer, groups)
        concept_matches += sum(matches)
        concept_total += len(matches)
        citations, citations_valid = _valid_citations(
            answer, len(observation.get("sources") or [])
        )
        if case["should_answer"]:
            answerable_cases += 1
            citation_passes += int(citations_valid)

        results.append(
            {
                "id": case["id"],
                "should_answer": case["should_answer"],
                "grounded": grounded,
                "generation_invoked": generation_invoked,
                "guardrail_correct": guardrail_correct,
                "concept_matches": matches,
                "concept_coverage": sum(matches) / len(matches) if matches else None,
                "citations": citations,
                "citations_valid": citations_valid if case["should_answer"] else None,
                "sources": observation.get("sources") or [],
                "answer": answer,
            }
        )

    metrics = {
        "concept_coverage": concept_matches / concept_total,
        "citation_accuracy": citation_passes / answerable_cases,
        "guardrail_accuracy": guardrail_passes / len(cases),
        "concept_matches": concept_matches,
        "concept_total": concept_total,
        "citation_passes": citation_passes,
        "answerable_cases": answerable_cases,
        "guardrail_passes": guardrail_passes,
        "total_cases": len(cases),
    }
    minimums = dataset["minimums"]
    passed = all(metrics[name] >= threshold for name, threshold in minimums.items())
    return {
        "dataset": dataset["name"],
        "minimums": minimums,
        "metrics": metrics,
        "passed": passed,
        "results": results,
    }


def print_report(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    print(f"Dataset: {report['dataset']} ({metrics['total_cases']} cases)")
    print(
        f"Concept coverage: {metrics['concept_coverage']:.1%} "
        f"({metrics['concept_matches']}/{metrics['concept_total']})"
    )
    print(
        f"Citation accuracy: {metrics['citation_accuracy']:.1%} "
        f"({metrics['citation_passes']}/{metrics['answerable_cases']})"
    )
    print(
        f"Guardrail accuracy: {metrics['guardrail_accuracy']:.1%} "
        f"({metrics['guardrail_passes']}/{metrics['total_cases']})"
    )
    print(f"Result: {'PASS' if report['passed'] else 'FAIL'}")
    print("\n| Case | Guardrail | Concepts | Citations |")
    print("|---|---|---:|---:|")
    for result in report["results"]:
        coverage = result["concept_coverage"]
        coverage_text = f"{coverage:.0%}" if coverage is not None else "-"
        citation = result["citations_valid"]
        citation_text = "PASS" if citation else "FAIL" if citation is False else "-"
        guardrail_text = "PASS" if result["guardrail_correct"] else "FAIL"
        print(
            f"| {result['id']} | {guardrail_text} | {coverage_text} | "
            f"{citation_text} |"
        )


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    dataset = load_dataset(dataset_path)
    if args.validate_only:
        print(f"Valid dataset: {dataset['name']} ({len(dataset['cases'])} cases)")
        return

    from app.rag import create_rag_service

    service = create_rag_service()
    observations = []
    for case in dataset["cases"]:
        print(f"Evaluating {case['id']}...", file=sys.stderr)
        result = service.query(case["question"], dataset["top_k"])
        observations.append(
            {
                "id": case["id"],
                "grounded": result.grounded,
                "generation_invoked": result.generation is not None,
                "sources": [chunk.source for chunk in result.chunks],
                "answer": result.answer,
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
