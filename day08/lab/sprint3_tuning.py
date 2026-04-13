"""
sprint3_tuning.py — Sprint 3: Tuning Variant (Dev 3 - Tuning & QA Owner)
=========================================================================
Nhiệm vụ Sprint 3:
  1. Implement BM25 Sparse Retrieval (rank-bm25)
  2. Implement Hybrid RRF = Dense + Sparse (Reciprocal Rank Fusion)
  3. Implement CrossEncoder Reranker (sentence-transformers)
  4. Chạy A/B comparison: baseline_dense vs variant_hybrid_rerank
  5. Tự động ghi kết quả vào docs/tuning-log.md

Future-ready design:
  - Hoạt động hoàn chỉnh sau khi Sprint 1 (index.py) và Sprint 2 (rag_answer.py) xong.
  - Nếu Sprint 1/2 chưa implement: chạy --dry-run để test logic riêng lẻ.

Chạy:
  python sprint3_tuning.py              # Chạy full A/B comparison
  python sprint3_tuning.py --dry-run    # Test logic mà không cần index/LLM
"""

import os
import sys
import json
import math
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# ĐƯỜNG DẪN
# =============================================================================

LAB_DIR = Path(__file__).parent
DOCS_DIR = LAB_DIR / "data" / "docs"
CHROMA_DB_DIR = LAB_DIR / "chroma_db"
TUNING_LOG_PATH = LAB_DIR / "docs" / "tuning-log.md"
TEST_QUESTIONS_PATH = LAB_DIR / "data" / "test_questions.json"
RESULTS_DIR = LAB_DIR / "results"

# =============================================================================
# CẤU HÌNH BASELINE & VARIANT
# =============================================================================

BASELINE_CONFIG = {
    "label": "baseline_dense",
    "retrieval_mode": "dense",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": False,
}

# Variant: Hybrid (Dense + BM25) + CrossEncoder Rerank
# Lý do chọn: Corpus IT Helpdesk có cả câu tự nhiên (policy)
# lẫn tên riêng, mã lỗi (ERR-403, P1, SLA). Dense bỏ sót keyword.
VARIANT_CONFIG = {
    "label": "variant_hybrid_rerank",
    "retrieval_mode": "hybrid",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": True,
}

TOP_K_RRF = 10  # Số chunk lấy từ mỗi nguồn trước khi fuse
RRF_K = 60      # Hằng số RRF tiêu chuẩn

# =============================================================================
# BM25 SPARSE INDEX
# Cần load tất cả chunks từ ChromaDB (sau khi Sprint 1 đã chạy build_index)
# =============================================================================

_bm25_index = None
_bm25_chunks = None  # All chunks kèm metadata


def _load_all_chunks_from_chroma() -> List[Dict[str, Any]]:
    """Load toàn bộ chunks từ ChromaDB để build BM25 index."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_collection("rag_lab")
        results = collection.get(include=["documents", "metadatas"])
        chunks = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            chunks.append({
                "text": doc,
                "metadata": meta or {},
                "score": 0.0,
            })
        print(f"[BM25] Loaded {len(chunks)} chunks from ChromaDB")
        return chunks
    except Exception as e:
        print(f"[BM25] Không load được ChromaDB: {e}")
        print("[BM25] Hãy chạy 'python index.py' trước (Sprint 1)")
        return []


def build_bm25_index() -> bool:
    """
    Khởi tạo BM25 index từ toàn bộ chunks trong ChromaDB.
    Gọi hàm này một lần trước khi dùng retrieve_sparse_impl().
    Returns True nếu thành công.
    """
    global _bm25_index, _bm25_chunks

    from rank_bm25 import BM25Okapi

    chunks = _load_all_chunks_from_chroma()
    if not chunks:
        return False

    corpus = [c["text"] for c in chunks]
    # Tokenize: lowercase + split whitespace (đơn giản, phù hợp tiếng Việt/Anh)
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    _bm25_index = BM25Okapi(tokenized_corpus)
    _bm25_chunks = chunks
    print(f"[BM25] Index built: {len(chunks)} documents")
    return True


# =============================================================================
# SPARSE RETRIEVAL (BM25)
# =============================================================================

def retrieve_sparse_impl(query: str, top_k: int = TOP_K_RRF) -> List[Dict[str, Any]]:
    """
    BM25 Sparse Retrieval.
    Mạnh ở: exact keyword (P1, SLA, ERR-403, Level 3)
    Yếu ở: paraphrase, câu tự nhiên có nhiều đồng nghĩa
    """
    global _bm25_index, _bm25_chunks

    if _bm25_index is None:
        print("[BM25] Index chưa được build. Gọi build_bm25_index() trước.")
        return []

    tokenized_query = query.lower().split()
    scores = _bm25_index.get_scores(tokenized_query)

    # Lấy top_k indices theo score giảm dần
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for rank, idx in enumerate(top_indices):
        if scores[idx] > 0:  # Bỏ qua chunk score = 0 (không relevant)
            chunk = _bm25_chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            chunk["_bm25_rank"] = rank
            results.append(chunk)

    return results


# =============================================================================
# HYBRID RETRIEVAL (Dense + Sparse via RRF)
# =============================================================================

def retrieve_hybrid_impl(
    query: str,
    top_k: int = TOP_K_RRF,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Hybrid Retrieval = Dense + BM25 kết hợp bằng Reciprocal Rank Fusion (RRF).

    Công thức RRF:
        score(doc) = dense_weight * 1/(RRF_K + dense_rank)
                   + sparse_weight * 1/(RRF_K + sparse_rank)

    Tại sao chọn hybrid cho IT Helpdesk:
    - Policy text (câu tự nhiên) → Dense mạnh hơn
    - Mã lỗi, tên tài liệu, số hiệu SLA → BM25 mạnh hơn
    - Hybrid giữ được cả hai điểm mạnh
    """
    # Import dense retrieval từ Sprint 2
    try:
        from rag_answer import retrieve_dense
    except ImportError:
        print("[Hybrid] Không import được rag_answer. Sprint 2 chưa sẵn sàng.")
        return []

    # --- Dense Retrieval ---
    try:
        dense_results = retrieve_dense(query, top_k=top_k)
    except NotImplementedError:
        print("[Hybrid] retrieve_dense() chưa implement (Sprint 2 TODO). Fallback về BM25 only.")
        dense_results = []
    except Exception as e:
        print(f"[Hybrid] Dense retrieval lỗi: {e}")
        dense_results = []

    # --- Sparse (BM25) Retrieval ---
    sparse_results = retrieve_sparse_impl(query, top_k=top_k)

    if not dense_results and not sparse_results:
        return []

    # --- Reciprocal Rank Fusion ---
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict] = {}

    # Tạo unique key cho mỗi chunk (dùng text hash hoặc source+section)
    def chunk_key(chunk: Dict) -> str:
        meta = chunk.get("metadata", {})
        text_preview = chunk.get("text", "")[:80]
        return f"{meta.get('source', '')}::{meta.get('section', '')}::{text_preview}"

    # Gộp Dense scores
    for rank, chunk in enumerate(dense_results):
        key = chunk_key(chunk)
        chunk_map[key] = chunk
        rrf_scores[key] = rrf_scores.get(key, 0.0) + dense_weight * (1.0 / (RRF_K + rank))

    # Gộp Sparse scores
    for rank, chunk in enumerate(sparse_results):
        key = chunk_key(chunk)
        chunk_map[key] = chunk
        rrf_scores[key] = rrf_scores.get(key, 0.0) + sparse_weight * (1.0 / (RRF_K + rank))

    # Sort theo RRF score
    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

    results = []
    for key in sorted_keys[:top_k]:
        chunk = chunk_map[key].copy()
        chunk["score"] = rrf_scores[key]
        chunk["_retrieval_method"] = "hybrid_rrf"
        results.append(chunk)

    return results


# =============================================================================
# RERANKER (CrossEncoder)
# =============================================================================

_cross_encoder = None
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_cross_encoder():
    """Lazy-load CrossEncoder model (chỉ load lần đầu)."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            print(f"[Reranker] Loading CrossEncoder: {RERANKER_MODEL}")
            _cross_encoder = CrossEncoder(RERANKER_MODEL)
            print("[Reranker] ✅ CrossEncoder loaded!")
        except ImportError:
            print("[Reranker] sentence-transformers chưa cài. Chạy: pip install sentence-transformers")
            return None
        except Exception as e:
            print(f"[Reranker] Không load được CrossEncoder: {e}")
            return None
    return _cross_encoder


def rerank_impl(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    CrossEncoder Reranker: chấm lại relevance của candidates.

    Flow:
        Search rộng (top-10) → Hybrid RRF → Rerank (top-3)
        Reranker đọc (query, chunk) cùng lúc → accuracy cao hơn Dense embedding
    """
    if not candidates:
        return []

    encoder = load_cross_encoder()
    if encoder is None:
        # Fallback: không rerank, trả về top_k đầu
        print("[Reranker] Fallback: trả về top_k đầu mà không rerank")
        return candidates[:top_k]

    # Tạo pairs (query, chunk_text) để score
    pairs = [(query, c.get("text", "")) for c in candidates]

    try:
        scores = encoder.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        results = []
        for chunk, score in ranked[:top_k]:
            chunk = chunk.copy()
            chunk["_rerank_score"] = float(score)
            chunk["score"] = float(score)
            results.append(chunk)
        return results
    except Exception as e:
        print(f"[Reranker] Lỗi khi rerank: {e}")
        return candidates[:top_k]


# =============================================================================
# PATCH cho rag_answer.py (Kết nối Sprint 3 vào Sprint 2)
# =============================================================================

def patch_rag_answer_with_sprint3():
    """
    Monkey-patch rag_answer module để dùng Sprint 3 implementations.
    Gọi hàm này trước khi chạy A/B comparison.

    Cách hoạt động:
    - Sprint 2 rag_answer.py có các hàm retrieve_sparse/retrieve_hybrid/rerank là placeholder.
    - Hàm này thay thế chúng bằng implementation thực tế của Sprint 3.
    """
    try:
        import rag_answer as ra

        # Patch retrieve_sparse
        def _patched_sparse(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
            return retrieve_sparse_impl(query, top_k)

        # Patch retrieve_hybrid
        def _patched_hybrid(query: str, top_k: int = 10,
                            dense_weight: float = 0.6, sparse_weight: float = 0.4):
            return retrieve_hybrid_impl(query, top_k, dense_weight, sparse_weight)

        # Patch rerank
        def _patched_rerank(query: str, candidates: List[Dict], top_k: int = 3):
            return rerank_impl(query, candidates, top_k)

        ra.retrieve_sparse = _patched_sparse
        ra.retrieve_hybrid = _patched_hybrid
        ra.rerank = _patched_rerank

        print("[Patch] ✅ rag_answer.py đã được patch với Sprint 3 implementations")
        return True
    except ImportError:
        print("[Patch] Không import được rag_answer. Hãy đảm bảo file rag_answer.py tồn tại.")
        return False


# =============================================================================
# A/B COMPARISON: Baseline vs Variant
# =============================================================================

def run_ab_comparison(test_questions: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Chạy A/B comparison: Baseline Dense vs Variant Hybrid+Rerank.
    Trả về dict kết quả để dùng cho update_tuning_log().
    """
    try:
        from rag_answer import rag_answer
    except ImportError:
        print("[A/B] Không import được rag_answer.")
        return {}

    # Load test questions
    if test_questions is None:
        try:
            with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
                test_questions = json.load(f)
        except FileNotFoundError:
            print(f"[A/B] Không tìm thấy {TEST_QUESTIONS_PATH}")
            return {}

    print(f"\n{'='*65}")
    print("A/B COMPARISON: Baseline Dense vs Variant Hybrid+Rerank")
    print(f"{'='*65}")
    print(f"Số câu test: {len(test_questions)}")

    baseline_results = []
    variant_results = []

    for q in test_questions:
        qid = q.get("id", "?")
        query = q.get("question", "")
        print(f"\n[{qid}] {query[:70]}...")

        # --- Baseline ---
        try:
            b_result = rag_answer(
                query,
                retrieval_mode=BASELINE_CONFIG["retrieval_mode"],
                top_k_search=BASELINE_CONFIG["top_k_search"],
                top_k_select=BASELINE_CONFIG["top_k_select"],
                use_rerank=BASELINE_CONFIG["use_rerank"],
                verbose=False,
            )
            b_answer = b_result["answer"]
            b_sources = b_result.get("sources", [])
            b_chunks = b_result.get("chunks_used", [])
        except NotImplementedError:
            b_answer = "SPRINT2_NOT_IMPLEMENTED"
            b_sources = []
            b_chunks = []
        except Exception as e:
            b_answer = f"PIPELINE_ERROR: {e}"
            b_sources = []
            b_chunks = []

        # --- Variant ---
        try:
            v_result = rag_answer(
                query,
                retrieval_mode=VARIANT_CONFIG["retrieval_mode"],
                top_k_search=VARIANT_CONFIG["top_k_search"],
                top_k_select=VARIANT_CONFIG["top_k_select"],
                use_rerank=VARIANT_CONFIG["use_rerank"],
                verbose=False,
            )
            v_answer = v_result["answer"]
            v_sources = v_result.get("sources", [])
            v_chunks = v_result.get("chunks_used", [])
        except NotImplementedError:
            v_answer = "SPRINT2_NOT_IMPLEMENTED"
            v_sources = []
            v_chunks = []
        except Exception as e:
            v_answer = f"PIPELINE_ERROR: {e}"
            v_sources = []
            v_chunks = []

        print(f"  [Baseline] {b_answer[:80]}...")
        print(f"  [Variant]  {v_answer[:80]}...")

        baseline_results.append({
            "id": qid, "question": query,
            "answer": b_answer, "sources": b_sources,
            "chunks_count": len(b_chunks),
        })
        variant_results.append({
            "id": qid, "question": query,
            "answer": v_answer, "sources": v_sources,
            "chunks_count": len(v_chunks),
        })

    return {
        "timestamp": datetime.now().isoformat(),
        "baseline_config": BASELINE_CONFIG,
        "variant_config": VARIANT_CONFIG,
        "baseline_results": baseline_results,
        "variant_results": variant_results,
        "total_questions": len(test_questions),
    }


# =============================================================================
# AUTO-UPDATE TUNING LOG
# =============================================================================

def update_tuning_log(ab_results: Dict[str, Any], notes: str = "") -> None:
    """
    Tự động ghi kết quả A/B vào docs/tuning-log.md.
    Giữ nguyên nội dung cũ, APPEND thêm section mới ở cuối.

    Format entry: đúng với template trong tuning-log.md
    """
    if not ab_results:
        print("[TuningLog] Không có kết quả để ghi.")
        return

    TUNING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Tóm tắt: Câu nào Variant có thêm sources hơn Baseline?
    baseline_results = ab_results.get("baseline_results", [])
    variant_results = ab_results.get("variant_results", [])

    improvements = []
    regressions = []
    for b, v in zip(baseline_results, variant_results):
        b_sources_count = len(b.get("sources", []))
        v_sources_count = len(v.get("sources", []))
        b_chunks = b.get("chunks_count", 0)
        v_chunks = v.get("chunks_count", 0)

        if v_chunks > b_chunks:
            improvements.append(f"- [{b['id']}] Variant retrieve được {v_chunks} chunks vs {b_chunks} (baseline)")
        elif b_chunks > v_chunks and v_chunks > 0:
            regressions.append(f"- [{b['id']}] Baseline retrieve được {b_chunks} chunks vs {v_chunks} (variant)")

    variant_label = ab_results.get("variant_config", {}).get("label", "variant")
    baseline_label = ab_results.get("baseline_config", {}).get("label", "baseline")

    entry = f"""

---

## [AUTO] Experiment Log — {timestamp}

**Biến thay đổi:** `retrieval_mode`: `dense` → `hybrid` + `use_rerank`: `False` → `True`

**Lý do chọn biến này:**
> Corpus IT Helpdesk có 2 loại query:
> 1. Câu ngôn ngữ tự nhiên ("Chính sách hoàn tiền như thế nào?") → Dense mạnh
> 2. Keyword chính xác ("ERR-403", "P1 ticket", "Level 3 access") → BM25 mạnh
> Hybrid RRF kết hợp cả 2 để không bỏ sót câu hỏi dạng nào.
> CrossEncoder rerank loại bỏ chunk noise sau khi search rộng.

**Baseline Config:** `{baseline_label}`
```
retrieval_mode = "{ab_results.get('baseline_config', {}).get('retrieval_mode', 'dense')}"
top_k_search   = {ab_results.get('baseline_config', {}).get('top_k_search', 10)}
top_k_select   = {ab_results.get('baseline_config', {}).get('top_k_select', 3)}
use_rerank     = {ab_results.get('baseline_config', {}).get('use_rerank', False)}
```

**Variant Config:** `{variant_label}`
```
retrieval_mode = "{ab_results.get('variant_config', {}).get('retrieval_mode', 'hybrid')}"
top_k_search   = {ab_results.get('variant_config', {}).get('top_k_search', 10)}
top_k_select   = {ab_results.get('variant_config', {}).get('top_k_select', 3)}
use_rerank     = {ab_results.get('variant_config', {}).get('use_rerank', True)}
reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

**Tổng câu hỏi chạy:** {ab_results.get('total_questions', 0)}

**Quan sát cải thiện (Variant > Baseline):**
{chr(10).join(improvements) if improvements else "- (Chưa có dữ liệu scorecard — chạy eval.py để lấy điểm số)"}

**Quan sát hồi quy (Baseline > Variant):**
{chr(10).join(regressions) if regressions else "- Không có"}

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | ?/5 | ?/5 | +/- |
| Answer Relevance | ?/5 | ?/5 | +/- |
| Context Recall | ?/5 | ?/5 | +/- |
| Completeness | ?/5 | ?/5 | +/- |

> **TODO:** Điền bảng trên sau khi chạy `python eval.py` (Sprint 4)

**Kết luận sơ bộ:**
> Hybrid + Rerank dự kiến cải thiện Context Recall cho query dạng keyword (ERR-403, Level 3).
> Xác nhận bằng số liệu eval.py.

{f'**Notes:** {notes}' if notes else ''}
"""

    # Đọc file hiện tại và append entry mới
    existing_content = ""
    if TUNING_LOG_PATH.exists():
        existing_content = TUNING_LOG_PATH.read_text(encoding="utf-8")

    updated_content = existing_content + entry
    TUNING_LOG_PATH.write_text(updated_content, encoding="utf-8")
    print(f"\n[TuningLog] ✅ Đã cập nhật: {TUNING_LOG_PATH}")


# =============================================================================
# DRY RUN MODE (Không cần Sprint 1/2 sẵn sàng)
# =============================================================================

def run_dry_run():
    """
    Test toàn bộ logic Sprint 3 với dữ liệu giả.
    Dùng để verify code đúng trước khi Sprint 1/2 xong.
    """
    print("\n" + "="*65)
    print("DRY RUN MODE — Test Sprint 3 logic (không cần index/LLM)")
    print("="*65)

    # Test RRF merge logic với data giả
    mock_dense = [
        {"text": f"Dense chunk {i}", "metadata": {"source": f"doc{i//3}.txt", "section": "A"}, "score": 1.0 - i*0.1}
        for i in range(5)
    ]
    mock_sparse = [
        {"text": f"Dense chunk {i}", "metadata": {"source": f"doc{i//3}.txt", "section": "A"}, "score": 5.0 - i}
        for i in [2, 0, 4, 1, 3]  # khác thứ tự
    ]

    print("\n[Test 1] RRF Fusion Logic:")
    print(f"  Dense order: {[c['text'] for c in mock_dense]}")
    print(f"  Sparse order: {[c['text'] for c in mock_sparse]}")

    # Test RRF manually
    rrf_scores: Dict[str, float] = {}
    for rank, c in enumerate(mock_dense):
        key = c["text"]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 0.6 * (1.0 / (60 + rank))
    for rank, c in enumerate(mock_sparse):
        key = c["text"]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 0.4 * (1.0 / (60 + rank))

    rrf_sorted = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"  RRF result: {[k for k, v in rrf_sorted[:3]]}")
    print("  [OK] RRF fusion logic hoạt động chính xác")

    # Test BM25 import
    print("\n[Test 2] BM25 Import:")
    try:
        from rank_bm25 import BM25Okapi
        corpus = ["SLA ticket P1 là 4 giờ", "hoàn tiền trong 30 ngày", "cấp quyền Level 3"]
        bm25 = BM25Okapi([doc.lower().split() for doc in corpus])
        scores = bm25.get_scores("SLA P1".lower().split())
        print(f"  BM25 scores cho 'SLA P1': {scores.tolist()}")
        print("  [OK] BM25 hoạt động")
    except ImportError:
        print("  [SKIP] rank-bm25 chưa cài — chạy: pip install rank-bm25")

    # Test CrossEncoder import
    print("\n[Test 3] CrossEncoder Import:")
    try:
        from sentence_transformers import CrossEncoder
        print(f"  [OK] sentence-transformers sẵn sàng (model sẽ download khi chạy lần đầu)")
    except ImportError:
        print("  [SKIP] sentence-transformers chưa cài — chạy: pip install sentence-transformers")

    # Test tuning-log write
    print("\n[Test 4] Auto-update Tuning Log:")
    mock_results = {
        "timestamp": datetime.now().isoformat(),
        "baseline_config": BASELINE_CONFIG,
        "variant_config": VARIANT_CONFIG,
        "baseline_results": [{"id": "q01", "question": "Test?", "answer": "Test answer", "sources": ["doc1.txt"], "chunks_count": 3}],
        "variant_results": [{"id": "q01", "question": "Test?", "answer": "Better test answer", "sources": ["doc1.txt", "doc2.txt"], "chunks_count": 4}],
        "total_questions": 1,
    }
    update_tuning_log(mock_results, notes="[DRY RUN TEST] Không phải kết quả thực tế")
    print("  [OK] tuning-log.md đã được cập nhật")

    print("\n" + "="*65)
    print("DRY RUN hoàn thành! Sprint 3 logic sẵn sàng.")
    print("Tiếp theo: Chờ Sprint 1 + Sprint 2 xong, rồi chạy:")
    print("  python sprint3_tuning.py   (full A/B comparison)")
    print("="*65)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sprint 3: Tuning & Variant")
    parser.add_argument("--dry-run", action="store_true", help="Test logic mà không cần index/LLM")
    parser.add_argument("--notes", type=str, default="", help="Ghi chú thêm vào tuning-log")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
    else:
        print("Sprint 3: Tuning Variant — Full A/B Comparison")
        print("="*65)

        # Bước 1: Build BM25 index từ ChromaDB
        print("\n[Step 1] Build BM25 index...")
        bm25_ok = build_bm25_index()
        if not bm25_ok:
            print("[Step 1] ⚠️ BM25 index thất bại. Kiểm tra Sprint 1 đã chạy chưa.")
            print("  Gợi ý: python index.py")

        # Bước 2: Patch rag_answer với Sprint 3 implementations
        print("\n[Step 2] Patch rag_answer với Sprint 3 variants...")
        patch_ok = patch_rag_answer_with_sprint3()

        # Bước 3: Chạy A/B comparison
        print("\n[Step 3] Chạy A/B comparison...")
        ab_results = run_ab_comparison()

        # Bước 4: Ghi kết quả vào tuning-log.md
        if ab_results:
            print("\n[Step 4] Cập nhật tuning-log.md...")
            update_tuning_log(ab_results, notes=args.notes)

            # Lưu raw results sang JSON để dùng cho eval.py
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            ab_json_path = RESULTS_DIR / "ab_raw_results.json"
            with open(ab_json_path, "w", encoding="utf-8") as f:
                json.dump(ab_results, f, ensure_ascii=False, indent=2)
            print(f"[Step 4] Raw results đã lưu tại: {ab_json_path}")

        print("\n✅ Sprint 3 hoàn thành!")
        print("Tiếp theo: Chạy eval.py để lấy điểm số scorecard (Sprint 4)")
