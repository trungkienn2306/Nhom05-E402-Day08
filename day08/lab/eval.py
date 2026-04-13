"""
eval.py — Sprint 4: Evaluation & Scorecard (Manual Scoring)
"""

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from rag_answer import rag_answer_ab

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TEST_QUESTIONS_PATH = Path(__file__).parent / "data" / "test_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"

# Sprint 2 baseline: dense retrieval only
BASELINE_CONFIG = {
    "mode": "baseline",
    "variant_retrieval_mode": "dense",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": False,
    "label": "baseline_dense",
}

# Sprint 3 variant: hybrid retrieval only (chỉ đổi 1 biến)
VARIANT_CONFIG = {
    "mode": "variant",
    "variant_retrieval_mode": "hybrid",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": False,
    "label": "variant_hybrid",
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def _is_abstain(answer: str) -> bool:
    text = (answer or "").lower()
    keywords = [
        "không đủ dữ liệu",
        "khong du du lieu",
        "không có thông tin",
        "khong co thong tin",
        "i do not know",
        "insufficient context",
        "không tìm thấy",
        "khong tim thay",
    ]
    return any(k in text for k in keywords)


def _citation_count(answer: str) -> int:
    return len(re.findall(r"\[\d+\]", answer or ""))


def _coverage_ratio(answer: str, reference: str, stop: Optional[Set[str]] = None) -> float:
    if stop is None:
        stop = set()
    ans_tokens = {t for t in _tokenize(answer) if len(t) >= 3 and t not in stop}
    ref_tokens = {t for t in _tokenize(reference) if len(t) >= 3 and t not in stop}
    if not ref_tokens:
        return 0.0
    return len(ans_tokens & ref_tokens) / len(ref_tokens)


def score_faithfulness(answer: str, chunks_used: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Manual rule-based faithfulness (không dùng LLM judge).
    """
    if not answer:
        return {"score": 1, "notes": "Empty answer"}
    if _is_abstain(answer):
        return {"score": 5, "notes": "Abstain response detected"}
    if not chunks_used:
        return {"score": 1, "notes": "No retrieved context"}

    context_text = " ".join(c.get("text", "") for c in chunks_used)
    ratio = _coverage_ratio(answer, context_text)
    cites = _citation_count(answer)

    if ratio >= 0.60:
        score = 5
    elif ratio >= 0.45:
        score = 4
    elif ratio >= 0.30:
        score = 3
    elif ratio >= 0.15:
        score = 2
    else:
        score = 1

    if cites > 0 and score < 5:
        score += 1
    return {"score": min(score, 5), "notes": f"coverage={ratio:.2f}, citations={cites}"}


def score_answer_relevance(query: str, answer: str) -> Dict[str, Any]:
    """
    Manual relevance based on lexical overlap between query and answer.
    """
    if not answer:
        return {"score": 1, "notes": "Empty answer"}
    if _is_abstain(answer):
        q = query.lower()
        if "err-403-auth" in q or "lỗi gì" in q:
            return {"score": 5, "notes": "Expected abstain-like query"}
        return {"score": 3, "notes": "Abstain without clear unknown query"}

    stop = {"ticket", "query", "câu", "hỏi", "what", "which", "bao", "nhiêu", "là", "thế", "nào"}
    ratio = _coverage_ratio(answer, query, stop)
    if ratio >= 0.80:
        score = 5
    elif ratio >= 0.60:
        score = 4
    elif ratio >= 0.40:
        score = 3
    elif ratio >= 0.20:
        score = 2
    else:
        score = 1
    return {"score": score, "notes": f"query_overlap={ratio:.2f}"}


def score_context_recall(chunks_used: List[Dict[str, Any]], expected_sources: List[str]) -> Dict[str, Any]:
    if not expected_sources:
        return {"score": None, "recall": None, "notes": "No expected sources"}

    retrieved_sources = {c.get("metadata", {}).get("source", "").lower() for c in chunks_used}
    found = 0
    missing = []

    for expected in expected_sources:
        expected_name = expected.split("/")[-1].replace(".pdf", "").replace(".md", "").lower()
        matched = any(expected_name in src for src in retrieved_sources)
        if matched:
            found += 1
        else:
            missing.append(expected)

    recall = found / len(expected_sources)
    score = max(1, round(recall * 5))
    return {
        "score": score,
        "recall": recall,
        "found": found,
        "missing": missing,
        "notes": f"Retrieved: {found}/{len(expected_sources)}",
    }


def score_completeness(query: str, answer: str, expected_answer: str) -> Dict[str, Any]:
    """
    Manual completeness based on expected answer coverage.
    """
    if not expected_answer:
        return {"score": 5, "notes": "No expected answer provided"}
    if not answer:
        return {"score": 1, "notes": "Empty answer"}
    if _is_abstain(answer):
        # Unknown query should abstain: treat as good completeness
        if "err-403-auth" in query.lower():
            return {"score": 5, "notes": "Expected abstain for unknown error code"}
        return {"score": 2, "notes": "Abstain but expected factual answer exists"}

    ratio = _coverage_ratio(answer, expected_answer)
    if ratio >= 0.85:
        score = 5
    elif ratio >= 0.65:
        score = 4
    elif ratio >= 0.45:
        score = 3
    elif ratio >= 0.25:
        score = 2
    else:
        score = 1
    return {"score": score, "notes": f"expected_coverage={ratio:.2f}"}


def run_scorecard(
    config: Dict[str, Any],
    test_questions: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    if test_questions is None:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)

    results = []
    label = config.get("label", "unnamed")
    print(f"\n{'=' * 70}\nChạy scorecard: {label}\nConfig: {config}\n{'=' * 70}")

    for q in test_questions:
        query = q["question"]
        expected_answer = q.get("expected_answer", "")
        expected_sources = q.get("expected_sources", [])
        if verbose:
            print(f"\n[{q['id']}] {query}")

        try:
            result = rag_answer_ab(
                query=query,
                mode=config.get("mode", "baseline"),
                variant_retrieval_mode=config.get("variant_retrieval_mode", "hybrid"),
                top_k_search=config.get("top_k_search", 10),
                top_k_select=config.get("top_k_select", 3),
                use_rerank=config.get("use_rerank", False),
                verbose=False,
            )
            answer = result["answer"]
            chunks_used = result["chunks_used"]
        except Exception as e:
            answer = f"ERROR: {e}"
            chunks_used = []

        faith = score_faithfulness(answer, chunks_used)
        relevance = score_answer_relevance(query, answer)
        recall = score_context_recall(chunks_used, expected_sources)
        complete = score_completeness(query, answer, expected_answer)

        row = {
            "id": q["id"],
            "category": q.get("category", ""),
            "query": query,
            "answer": answer,
            "expected_answer": expected_answer,
            "faithfulness": faith["score"],
            "faithfulness_notes": faith["notes"],
            "relevance": relevance["score"],
            "relevance_notes": relevance["notes"],
            "context_recall": recall["score"],
            "context_recall_notes": recall["notes"],
            "completeness": complete["score"],
            "completeness_notes": complete["notes"],
            "config_label": label,
        }
        results.append(row)
        if verbose:
            print(
                f"  Faithful: {faith['score']} | Relevant: {relevance['score']} | "
                f"Recall: {recall['score']} | Complete: {complete['score']}"
            )

    for metric in ["faithfulness", "relevance", "context_recall", "completeness"]:
        scores = [r[metric] for r in results if r[metric] is not None]
        avg = sum(scores) / len(scores) if scores else None
        print(f"Average {metric}: {avg:.2f}" if avg is not None else f"Average {metric}: N/A")
    return results


def compare_ab(baseline_results: List[Dict[str, Any]], variant_results: List[Dict[str, Any]], output_csv: Optional[str] = None) -> None:
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
    print(f"\n{'=' * 70}\nA/B Comparison: Baseline vs Variant\n{'=' * 70}")
    print(f"{'Metric':<20} {'Baseline':>10} {'Variant':>10} {'Delta':>8}")
    print("-" * 55)

    for metric in metrics:
        b = [r[metric] for r in baseline_results if r[metric] is not None]
        v = [r[metric] for r in variant_results if r[metric] is not None]
        b_avg = sum(b) / len(b) if b else None
        v_avg = sum(v) / len(v) if v else None
        delta = (v_avg - b_avg) if (b_avg is not None and v_avg is not None) else None
        print(
            f"{metric:<20} "
            f"{(f'{b_avg:.2f}' if b_avg is not None else 'N/A'):>10} "
            f"{(f'{v_avg:.2f}' if v_avg is not None else 'N/A'):>10} "
            f"{(f'{delta:+.2f}' if delta is not None else 'N/A'):>8}"
        )

    if output_csv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / output_csv
        combined = baseline_results + variant_results
        if combined:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=combined[0].keys())
                writer.writeheader()
                writer.writerows(combined)
            print(f"\nKết quả đã lưu vào: {csv_path}")


def generate_scorecard_summary(results: List[Dict[str, Any]], label: str) -> str:
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
    averages: Dict[str, Optional[float]] = {}
    for metric in metrics:
        vals = [r[metric] for r in results if r[metric] is not None]
        averages[metric] = (sum(vals) / len(vals)) if vals else None

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = [
        f"# Scorecard: {label}",
        f"Generated: {timestamp}",
        "",
        "## Summary",
        "",
        "| Metric | Average Score |",
        "|--------|--------------|",
    ]
    for metric, avg in averages.items():
        md.append(f"| {metric.replace('_', ' ').title()} | {f'{avg:.2f}/5' if avg is not None else 'N/A'} |")

    md.extend(
        [
            "",
            "## Per-Question Results",
            "",
            "| ID | Category | Faithful | Relevant | Recall | Complete | Notes |",
            "|----|----------|----------|----------|--------|----------|-------|",
        ]
    )
    for r in results:
        md.append(
            f"| {r['id']} | {r['category']} | {r.get('faithfulness', 'N/A')} | "
            f"{r.get('relevance', 'N/A')} | {r.get('context_recall', 'N/A')} | "
            f"{r.get('completeness', 'N/A')} | {r.get('faithfulness_notes', '')[:50]} |"
        )
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 4: Evaluation & Scorecard (Manual)")
    print("=" * 60)

    with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        test_questions = json.load(f)
    print(f"\nTìm thấy {len(test_questions)} câu hỏi trong {TEST_QUESTIONS_PATH}")

    baseline_results = run_scorecard(BASELINE_CONFIG, test_questions=test_questions, verbose=True)
    variant_results = run_scorecard(VARIANT_CONFIG, test_questions=test_questions, verbose=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "scorecard_baseline.md").write_text(
        generate_scorecard_summary(baseline_results, BASELINE_CONFIG["label"]),
        encoding="utf-8",
    )
    (RESULTS_DIR / "scorecard_variant.md").write_text(
        generate_scorecard_summary(variant_results, VARIANT_CONFIG["label"]),
        encoding="utf-8",
    )

    compare_ab(baseline_results, variant_results, output_csv="ab_comparison.csv")
