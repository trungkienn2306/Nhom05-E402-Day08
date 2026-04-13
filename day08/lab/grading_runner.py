"""
grading_runner.py — Chạy Grading Questions & Đảm bảo 10/10 Log
================================================================
Mục tiêu:
  - Đọc grading_questions.json (public lúc 17:00)
  - Chạy TOÀN BỘ 10 câu qua pipeline tốt nhất của nhóm
  - Xuất ra logs/grading_run.json đúng format SCORING.md
  - Đảm bảo KHÔNG thiếu câu → Lấy +1 điểm bonus
  - Xử lý crash gracefully: câu lỗi vẫn được ghi với PIPELINE_ERROR

Cách chạy:
  python grading_runner.py                    # Chạy full grading
  python grading_runner.py --test             # Test với 1 câu mock (không cần API)
  python grading_runner.py --mode dense       # Dùng baseline (dense)
  python grading_runner.py --mode hybrid      # Dùng variant (hybrid+rerank)
  python grading_runner.py --validate-log     # Kiểm tra log đã có

QUAN TRỌNG:
  - Chạy lệnh này sau 17:00 khi có file grading_questions.json
  - Phải commit log trước 18:00
"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# ĐƯỜNG DẪN
# =============================================================================

LAB_DIR = Path(__file__).parent
GRADING_Q_PATH = LAB_DIR / "data" / "grading_questions.json"
LOGS_DIR = LAB_DIR / "logs"
LOG_OUTPUT_PATH = LOGS_DIR / "grading_run.json"

# =============================================================================
# CẤU HÌNH PIPELINE TỐT NHẤT (Điều chỉnh sau khi Sprint 3 xong)
# =============================================================================

# Mặc định: dùng Hybrid + Rerank (variant tốt nhất)
DEFAULT_MODE = os.getenv("GRADING_MODE", "hybrid")
DEFAULT_USE_RERANK = True


# =============================================================================
# HÀM CHẠY 1 CÂU HỎI QUA PIPELINE
# =============================================================================

def run_single_question(
    q: Dict[str, Any],
    retrieval_mode: str = DEFAULT_MODE,
    use_rerank: bool = DEFAULT_USE_RERANK,
    top_k_search: int = 10,
    top_k_select: int = 3,
) -> Dict[str, Any]:
    """
    Chạy 1 câu hỏi qua RAG pipeline và trả về kết quả đúng format grading_run.json.

    Xử lý mọi loại lỗi:
    - NotImplementedError: Sprint 1/2 chưa implement → ghi SPRINT_NOT_IMPLEMENTED
    - Exception: crash bất kỳ → ghi PIPELINE_ERROR: [mô tả]
    - Không bao giờ để câu bị thiếu trong log (đảm bảo 10/10)
    """
    qid = q.get("id", "unknown")
    question = q.get("question", "")
    timestamp = datetime.now().isoformat(timespec="seconds")

    log_entry = {
        "id": qid,
        "question": question,
        "answer": "",
        "sources": [],
        "chunks_retrieved": 0,
        "retrieval_mode": retrieval_mode,
        "timestamp": timestamp,
        "status": "ok",
    }

    try:
        from rag_answer import rag_answer

        result = rag_answer(
            query=question,
            retrieval_mode=retrieval_mode,
            top_k_search=top_k_search,
            top_k_select=top_k_select,
            use_rerank=use_rerank,
            verbose=False,
        )

        log_entry["answer"] = result.get("answer", "")
        log_entry["sources"] = result.get("sources", [])
        log_entry["chunks_retrieved"] = len(result.get("chunks_used", []))
        log_entry["retrieval_mode"] = result.get("config", {}).get("retrieval_mode", retrieval_mode)
        log_entry["status"] = "ok"

    except NotImplementedError as e:
        err_msg = f"PIPELINE_ERROR: Sprint chưa implement — {str(e)[:200]}"
        print(f"  [⚠️ NotImplemented] {qid}: {err_msg[:80]}")
        log_entry["answer"] = err_msg
        log_entry["status"] = "not_implemented"

    except ImportError as e:
        err_msg = f"PIPELINE_ERROR: Import lỗi — {str(e)[:200]}"
        print(f"  [⚠️ ImportError] {qid}: {err_msg[:80]}")
        log_entry["answer"] = err_msg
        log_entry["status"] = "import_error"

    except Exception as e:
        err_msg = f"PIPELINE_ERROR: {type(e).__name__} — {str(e)[:200]}"
        print(f"  [❌ Error] {qid}: {err_msg[:80]}")
        traceback.print_exc()
        log_entry["answer"] = err_msg
        log_entry["status"] = "error"

    return log_entry


# =============================================================================
# CHẠY TOÀN BỘ GRADING QUESTIONS
# =============================================================================

def run_grading(
    grading_path: Path = GRADING_Q_PATH,
    output_path: Path = LOG_OUTPUT_PATH,
    retrieval_mode: str = DEFAULT_MODE,
    use_rerank: bool = DEFAULT_USE_RERANK,
) -> List[Dict[str, Any]]:
    """
    Hàm chính: Load grading_questions.json → Chạy pipeline → Xuất logs/grading_run.json.

    ĐẢM BẢO:
    - Mọi câu đều có entry trong log (kể cả câu lỗi)
    - Format đúng theo SCORING.md
    - Timestamp hợp lệ (17:00-18:00 để lấy bonus)
    """
    print("\n" + "="*65)
    print("GRADING RUNNER — Lab Day 08")
    print("="*65)
    print(f"Mode      : {retrieval_mode} | Rerank: {use_rerank}")
    print(f"Questions : {grading_path}")
    print(f"Output    : {output_path}")
    print(f"Timestamp : {datetime.now().strftime('%H:%M:%S')}")

    # Load grading questions
    if not grading_path.exists():
        print(f"\n❌ Không tìm thấy: {grading_path}")
        print("Grading questions sẽ được public lúc 17:00. Chờ đến khi có file!")
        return []

    with open(grading_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"\nTìm thấy {len(questions)} câu hỏi grading.\n")

    # Apply Sprint 3 patch nếu có
    _try_apply_sprint3_patch(retrieval_mode)

    # Chạy từng câu
    log = []
    for i, q in enumerate(questions, 1):
        qid = q.get("id", f"q{i:02d}")
        question_short = q.get("question", "")[:60]
        print(f"[{i:2d}/{len(questions)}] ({qid}) {question_short}...")

        entry = run_single_question(
            q,
            retrieval_mode=retrieval_mode,
            use_rerank=use_rerank,
        )
        log.append(entry)

        # In kết quả ngắn
        status_icon = "✅" if entry["status"] == "ok" else "⚠️"
        answer_preview = entry["answer"][:80].replace("\n", " ")
        print(f"      {status_icon} [{entry['status']}] {answer_preview}...")
        print(f"         Sources: {entry['sources'][:3]}, Chunks: {entry['chunks_retrieved']}")

    # Kiểm tra completeness
    print(f"\n{'='*65}")
    total = len(questions)
    ok_count = sum(1 for e in log if e["status"] == "ok")
    error_count = total - ok_count

    print(f"📊 Kết quả:")
    print(f"   ✅ Thành công : {ok_count}/{total}")
    print(f"   ❌ Lỗi       : {error_count}/{total}")

    if len(log) == total:
        print(f"   📋 Log đầy đủ: {len(log)}/{total} câu ← ĐỦ ĐỂ LẤY BONUS +1đ")
    else:
        print(f"   ⚠️ Log thiếu: {len(log)}/{total} câu — KHÔNG lấy được bonus!")

    # Xuất file log
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Log đã lưu tại: {output_path}")
    print(f"   Commit file này trước 18:00!\n")
    return log


# =============================================================================
# VALIDATE LOG ĐÃ CÓ
# =============================================================================

def validate_log(log_path: Path = LOG_OUTPUT_PATH) -> None:
    """
    Kiểm tra file log đã tạo có đúng format và đầy đủ không.
    Dùng để double-check trước khi commit.
    """
    print("\n" + "="*65)
    print("VALIDATE LOG")
    print("="*65)

    if not log_path.exists():
        print(f"❌ Không tìm thấy: {log_path}")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        log = json.load(f)

    print(f"Tổng entries: {len(log)}")

    required_fields = ["id", "question", "answer", "sources", "chunks_retrieved",
                       "retrieval_mode", "timestamp"]
    all_ok = True

    problems = []
    for i, entry in enumerate(log):
        # Check required fields
        for field in required_fields:
            if field not in entry:
                problems.append(f"Entry {i} ({entry.get('id','?')}): thiếu field '{field}'")
                all_ok = False

        # Check answer không rỗng
        if not entry.get("answer", "").strip():
            problems.append(f"Entry {i} ({entry.get('id','?')}): answer rỗng!")
            all_ok = False

        # Check timestamp hợp lý
        try:
            ts = datetime.fromisoformat(entry.get("timestamp", ""))
            if ts.hour < 17 or ts.hour >= 18:
                problems.append(f"Entry {i} ({entry.get('id','?')}): timestamp {ts.strftime('%H:%M')} ngoài 17:00-18:00 → MẤT bonus!")
        except (ValueError, TypeError):
            problems.append(f"Entry {i} ({entry.get('id','?')}): timestamp không hợp lệ")

    if problems:
        print("\n⚠️ Vấn đề phát hiện:")
        for p in problems:
            print(f"   - {p}")
    else:
        print("✅ Log hợp lệ! Tất cả fields đầy đủ.")

    # Summary per entry
    print("\nTóm tắt từng câu:")
    print(f"{'ID':<8} {'Status':<16} {'Chunks':<8} {'Mode':<12} {'Timestamp':<20}")
    print("-"*70)
    for e in log:
        status = e.get("status", "unknown")
        icon = "✅" if status == "ok" else "⚠️"
        print(f"{icon} {e.get('id','?'):<6} {status:<16} "
              f"{str(e.get('chunks_retrieved',0)):<8} "
              f"{e.get('retrieval_mode','?'):<12} "
              f"{e.get('timestamp','?')[:19]:<20}")

    # Bonus check
    if len(log) == 10 and all_ok:
        print("\n🎯 ĐỦ ĐIỀU KIỆN BONUS +1 ĐIỂM (10/10 câu, đúng format)")
    elif len(log) == 10:
        print(f"\n⚠️ Có {len(log)} câu nhưng có vấn đề format — kiểm tra lại")
    else:
        print(f"\n❌ Chỉ có {len(log)} câu — CẦN 10 câu để lấy bonus!")


# =============================================================================
# TEST MODE (Không cần Sprint 1/2 hay API key)
# =============================================================================

def run_test_mode() -> None:
    """
    Chạy test với mock data để verify grading runner hoạt động đúng.
    Không cần Sprint 1/2 hay API key.
    """
    print("\n" + "="*65)
    print("TEST MODE — Verify Grading Runner Logic")
    print("="*65)

    # Tạo mock grading questions
    mock_questions = [
        {"id": "gq01", "question": "SLA xử lý ticket P1 là bao lâu?"},
        {"id": "gq02", "question": "Remote access cần những điều kiện gì?"},
        {"id": "gq03", "question": "Flash Sale refund có được không?"},
    ]

    print(f"\nTest với {len(mock_questions)} câu mock (không cần API)")

    # Monkey-patch rag_answer để không cần Sprint 1/2
    class MockResult:
        pass

    import sys
    import types

    # Tạo mock module nếu rag_answer chưa implement
    try:
        from rag_answer import rag_answer
        print("[Test] rag_answer.py existed — using real module (may raise NotImplementedError)")
    except ImportError:
        print("[Test] rag_answer.py not found — creating mock")

    log = []
    for q in mock_questions:
        print(f"\nChạy: [{q['id']}] {q['question']}")

        # Thử gọi thực tế, catch mọi lỗi
        entry = run_single_question(q, retrieval_mode="hybrid", use_rerank=True)
        log.append(entry)
        print(f"  Status: {entry['status']} | Answer: {entry['answer'][:60]}...")

    # Test xuất file
    test_output = LOGS_DIR / "grading_run_TEST.json"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(test_output, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Test output lưu tại: {test_output}")
    print("Test Mode hoàn thành! Logic Grading Runner hoạt động đúng.")


# =============================================================================
# APPLY SPRINT 3 PATCH
# =============================================================================

def _try_apply_sprint3_patch(retrieval_mode: str) -> None:
    """Áp dụng Sprint 3 hybrid/rerank patch nếu có thể."""
    if retrieval_mode in ("hybrid",):
        try:
            from sprint3_tuning import patch_rag_answer_with_sprint3, build_bm25_index
            build_bm25_index()
            patch_rag_answer_with_sprint3()
        except ImportError:
            pass  # sprint3_tuning chưa có hoặc imports chưa sẵn sàng
        except Exception as e:
            print(f"[Sprint3 Patch] Không patch được: {e}. Tiếp tục với mode gốc.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grading Runner: Chạy 10 câu hỏi grading và xuất log chuẩn"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: chạy với mock data, không cần API key"
    )
    parser.add_argument(
        "--validate-log", action="store_true",
        help="Kiểm tra log đã tạo có đúng format không"
    )
    parser.add_argument(
        "--mode", choices=["dense", "hybrid", "sparse"], default=DEFAULT_MODE,
        help=f"Retrieval mode (mặc định: {DEFAULT_MODE})"
    )
    parser.add_argument(
        "--no-rerank", action="store_true",
        help="Tắt CrossEncoder rerank"
    )
    parser.add_argument(
        "--output", type=str, default=str(LOG_OUTPUT_PATH),
        help="Đường dẫn file output"
    )
    args = parser.parse_args()

    if args.test:
        run_test_mode()
    elif args.validate_log:
        validate_log(Path(args.output))
    else:
        run_grading(
            retrieval_mode=args.mode,
            use_rerank=not args.no_rerank,
            output_path=Path(args.output),
        )
