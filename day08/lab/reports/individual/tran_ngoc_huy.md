# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Trần Ngọc Huy
**Vai trò trong nhóm:** Tech Lead / Indexing Owner (Sprint 1)
**Ngày nộp:** 13/04/2026
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Tôi đảm nhận toàn bộ **Sprint 1 — Build RAG Index**, chịu trách nhiệm xây dựng nền tảng dữ liệu cho toàn bộ pipeline RAG.

Cụ thể, tôi đã implement 4 phần chính trong file `index.py`:

- **`get_embedding()`**: Tích hợp OpenAI API với model `text-embedding-3-small` để chuyển text thành vector 1536 chiều.
- **`_split_by_size()`**: Cải tiến logic cắt chunk từ "cắt cứng theo ký tự" sang "tìm ranh giới tự nhiên" (dấu chấm, xuống dòng, dấu cách) và thêm `.strip()` để loại khoảng trắng thừa.
- **`build_index()`**: Lắp ráp toàn bộ pipeline: đọc 5 file `.txt` → preprocess → chunk → embed → lưu vào ChromaDB với `upsert` và cosine similarity.
- **Phần `__main__`**: Uncomment `build_index()`, `list_chunks()`, `inspect_metadata_coverage()` để chạy end-to-end.

Công việc của tôi là **đầu vào trực tiếp** cho Sprint 2 và Sprint 3 — nếu ChromaDB chưa có data, không ai có thể retrieve hay evaluate được.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau Sprint 1, tôi hiểu rõ hơn về **chiến lược chunking và tầm quan trọng của ranh giới tự nhiên**.

Trước đây tôi nghĩ cắt text theo số ký tự cố định là đủ, nhưng thực tế khi đọc output của `list_chunks()`, tôi thấy các chunk bị cắt giữa câu sẽ mất ngữ cảnh hoàn toàn — ví dụ một điều khoản bị cắt giữa chừng thì model không thể trả lời chính xác dù retrieve đúng chunk.

Tôi cũng hiểu sâu hơn về **thiết kế 2 tầng**: tầng 1 ưu tiên cắt theo heading `=== Section ===` để giữ ranh giới ngữ nghĩa, tầng 2 mới fallback sang cắt theo kích thước. Điều này phản ánh nguyên tắc quan trọng: **RAG quality = chunking quality + retrieval quality**, không thể bù trừ cho nhau. Một chunk tốt sẽ giúp model trả lời đúng ngay cả khi retrieval không hoàn hảo.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Điều tôi không ngờ nhất là **môi trường Python** tốn nhiều thời gian hơn tôi nghĩ.

Ban đầu tôi chạy thẳng `python index.py` trong PowerShell và nhận được lỗi `Python was not found` dù đã cài PyCharm. Hóa ra terminal ngoài không nhận diện Python của PyCharm — phải activate đúng virtual environment `.venv` bằng lệnh `& ".venv\Scripts\Activate.ps1"`.

Ngoài ra, tôi gặp lỗi `chromadb.errors.NotFoundError` khi thử `delete_collection("rag_lab")` lần đầu chạy. Hướng dẫn bắt `except ValueError` nhưng phiên bản ChromaDB mới raise `NotFoundError` — đây là **breaking change** giữa các phiên bản thư viện. Bài học: không nên bắt exception quá hẹp, `except Exception` an toàn hơn trong trường hợp này.

Tổng cộng mất khoảng 20 phút chỉ để setup môi trường thay vì code — đây là lý do tại sao setup môi trường cần được làm sớm nhất trong mỗi sprint.

---

## 4. Phân tích một câu hỏi trong grading_questions (150-200 từ)

**Câu hỏi:** `gq07` — *"Công ty sẽ phạt bao nhiêu nếu team IT vi phạm cam kết SLA P1?"*

**Kết quả thực tế từ `logs/grading_run.json`:**

| Trường | Giá trị |
|--------|---------|
| **Answer** | `"Tôi không biết."` |
| **Sources retrieved** | `support/helpdesk-faq.md` |
| **Chunks retrieved** | 3 |
| **Status** | `ok` |

**Phân tích:**

Đây là câu hỏi **abstain** — thông tin về mức phạt vi phạm SLA không tồn tại trong bất kỳ tài liệu nào được index. Pipeline trả lời đúng với "Tôi không biết" → đạt **Full marks (10/10)** theo rubric SCORING.md.

Điều thú vị là pipeline **đã retrieve được 3 chunks** từ `helpdesk-faq.md` (tài liệu gần nhất về SLA), nhưng vẫn abstain đúng. Điều này chứng minh cả 3 tầng hoạt động chính xác:

1. **Sprint 1 — Indexing**: Chunking chỉ lưu đúng nội dung trong tài liệu, không "bịa" thêm thông tin về mức phạt.
2. **Sprint 2 — Generation**: Grounded prompt với quy tắc *"If context is insufficient, say you do not know"* hoạt động đúng — LLM không dùng kiến thức ngoài context.
3. **Sprint 3 — Retrieval**: Hybrid retrieval kéo về chunks liên quan nhất nhưng không đủ để trả lời → abstain, không hallucinate.

Đây là minh chứng quan trọng nhất cho giá trị của RAG có kiểm soát: **biết mình không biết còn quan trọng hơn trả lời sai.**

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

**Bằng chứng 1 — `logs/grading_run.json`, câu gq08:**
Câu hỏi thuần HR (nghỉ phép/nghỉ ốm) nhưng pipeline retrieve cả `it/access-control-sop.md` — sai domain hoàn toàn. Pattern tương tự lặp lại ở gq05 (contractor access) cũng kéo về `helpdesk-faq.md`. Tôi sẽ thêm **metadata pre-filter theo `department`** trước khi query ChromaDB: query chứa từ khoá HR ("nghỉ phép", "remote", "thử việc") → chỉ search `department=HR`; query về access/quyền → `department=IT Security`.

**Bằng chứng 2 — `results/ab_raw_results.json`, câu q06 Variant:**
CrossEncoder rerank đưa chunk "cấp quyền tạm thời 24h" (từ `access-control-sop.md`) vào câu trả lời về P1 escalation — làm lẫn 2 quy trình không liên quan. Tôi sẽ thêm **source-consistency check**: nếu top-3 chunks đến từ nhiều hơn 1 file, giữ lại file có tổng rerank score cao nhất.

