"""
eval.py — Sprint 4: Evaluation & Scorecard (Manual Scoring)
"""

import csv
import json
import os
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

# Bật LLM-as-Judge bằng biến môi trường:
#   USE_LLM_JUDGE=1  -> dùng judge LLM cho faithfulness/relevance/completeness
#   USE_LLM_JUDGE=0  -> dùng manual scoring (mặc định hiện tại)
USE_LLM_JUDGE = os.getenv("USE_LLM_JUDGE", "0").strip().lower() in {"1", "true", "yes", "y"}


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


def _safe_int_score(value: Any, default: int = 3) -> int:
    try:
        score = int(value)
    except Exception:
        return default
    return max(1, min(5, score))


def _extract_json_obj(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            text = m.group(0)
    return json.loads(text)


def _judge_with_llm(prompt: str, score_key: str = "score", reason_key: str = "reason") -> Dict[str, Any]:
    from rag_answer import call_llm

    try:
        raw = call_llm(prompt)
        obj = _extract_json_obj(raw)
        return {
            "score": _safe_int_score(obj.get(score_key)),
            "notes": str(obj.get(reason_key, "LLM judge")),
        }
    except Exception:
        return {"score": 3, "notes": "LLM judge parse error"}


def score_faithfulness(answer: str, chunks_used: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Manual rule-based faithfulness (không dùng LLM judge).
    """
    if USE_LLM_JUDGE:
        context_text = "\n\n".join(c.get("text", "") for c in chunks_used[:5])
        prompt = f"""
You are a strict RAG evaluator.
Evaluate FAITHFULNESS of the answer against retrieved context.

Answer:
{answer}

Retrieved context:
{context_text}

Scoring rubric (1-5):
- 5 = fully grounded, no hallucination
- 1 = mostly unsupported/hallucinated

Return ONLY JSON:
{{"score": 1-5, "reason": "short reason"}}
"""
        return _judge_with_llm(prompt, "score", "reason")

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
    if USE_LLM_JUDGE:
        prompt = f"""
You are a strict RAG evaluator.
Evaluate ANSWER RELEVANCE for the given question.

Question:
{query}

Answer:
{answer}

Scoring rubric (1-5):
- 5 = directly and clearly answers the question
- 1 = irrelevant / does not answer

Return ONLY JSON:
{{"score": 1-5, "reason": "short reason"}}
"""
        return _judge_with_llm(prompt, "score", "reason")

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
    if USE_LLM_JUDGE:
        prompt = f"""
You are a strict RAG evaluator.
Evaluate COMPLETENESS by comparing model answer vs expected answer.

Question:
{query}

Expected answer:
{expected_answer}

Model answer:
{answer}

Scoring rubric (1-5):
- 5 = covers all key points
- 1 = misses most key points

Return ONLY JSON:
{{"score": 1-5, "missing_points": "short note"}}
"""
        return _judge_with_llm(prompt, "score", "missing_points")

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


def compare_ab(
    baseline_results: List[Dict],
    variant_results: List[Dict],
    output_csv: Optional[str] = None,
) -> None:
    """
    So sánh baseline vs variant theo từng câu hỏi và tổng thể.

    TODO Sprint 4:
    Điền vào bảng sau để trình bày trong báo cáo:

    | Metric          | Baseline | Variant | Delta |
    |-----------------|----------|---------|-------|
    | Faithfulness    |   ?/5    |   ?/5   |  +/?  |
    | Answer Relevance|   ?/5    |   ?/5   |  +/?  |
    | Context Recall  |   ?/5    |   ?/5   |  +/?  |
    | Completeness    |   ?/5    |   ?/5   |  +/?  |

    Câu hỏi cần trả lời:
    - Variant tốt hơn baseline ở câu nào? Vì sao?
    - Biến nào (chunking / hybrid / rerank) đóng góp nhiều nhất?
    - Có câu nào variant lại kém hơn baseline không? Tại sao?
    """
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]

    print(f"\n{'='*70}")
    print("A/B Comparison: Baseline vs Variant")
    print('='*70)
    print(f"{'Metric':<20} {'Baseline':>10} {'Variant':>10} {'Delta':>8}")
    print("-" * 55)

    for metric in metrics:
        b_scores = [r[metric] for r in baseline_results if r[metric] is not None]
        v_scores = [r[metric] for r in variant_results if r[metric] is not None]

        b_avg = sum(b_scores) / len(b_scores) if b_scores else None
        v_avg = sum(v_scores) / len(v_scores) if v_scores else None
        delta = (v_avg - b_avg) if (b_avg and v_avg) else None

        b_str = f"{b_avg:.2f}" if b_avg else "N/A"
        v_str = f"{v_avg:.2f}" if v_avg else "N/A"
        d_str = f"{delta:+.2f}" if delta else "N/A"

        print(f"{metric:<20} {b_str:>10} {v_str:>10} {d_str:>8}")

    # Per-question comparison
    print(f"\n{'Câu':<6} {'Baseline F/R/Rc/C':<22} {'Variant F/R/Rc/C':<22} {'Better?':<10}")
    print("-" * 65)

    b_by_id = {r["id"]: r for r in baseline_results}
    for v_row in variant_results:
        qid = v_row["id"]
        b_row = b_by_id.get(qid, {})

        b_scores_str = "/".join([
            str(b_row.get(m, "?")) for m in metrics
        ])
        v_scores_str = "/".join([
            str(v_row.get(m, "?")) for m in metrics
        ])

        # So sánh đơn giản
        b_total = sum(b_row.get(m, 0) or 0 for m in metrics)
        v_total = sum(v_row.get(m, 0) or 0 for m in metrics)
        better = "Variant" if v_total > b_total else ("Baseline" if b_total > v_total else "Tie")

        print(f"{qid:<6} {b_scores_str:<22} {v_scores_str:<22} {better:<10}")

    # Export to CSV
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
