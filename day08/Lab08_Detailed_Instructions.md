# Hướng Dẫn Chi Tiết Thực Hành: Lab Day 08 — Full RAG Pipeline

## 1. Môn học và Nội dung Lab
- **Môn học:** AI in Action (AICB-P1)
- **Chủ đề:** RAG Pipeline: Xây dựng hệ thống trải qua 4 giai đoạn Indexing → Retrieval → Generation → Evaluation.
- **Mục tiêu chính:** Xây dựng một **trợ lý AI nội bộ cho khối CS + IT Helpdesk**. Trợ lý này sẽ chịu trách nhiệm trả lời các câu hỏi về chính sách hợp đồng, thời gian SLA xử lý ticket, quyền truy cập hệ thống và các FAQ khác dựa trên các tài liệu nội bộ, nhằm tránh tình trạng bịa đặt thông tin (hallucination).

---

## 2. Checklist Công Việc Thực Hành (4 Sprints)

Thời lượng thực hành chia làm 4 Sprint, mỗi Sprint 60 phút:

- [ ] **Sprint 1 — Build Index (Xây dựng vector database):**
  - *Code (`index.py`):* Tạo embeddings bằng OpenAI hoặc Sentence Transformers.
  - *Data:* Xử lý (chunking) 5 tài liệu văn bản cung cấp sẵn và lưu lên ChromaDB với ít nhất 3 loại metadata (`source`, `section`, `effective_date`).
  - *Kiểm tra:* Chạy được script lập chỉ mục, kiểm tra các câu (chunks) không bị cắt ngang giữa các đoạn tài liệu mang ý nghĩa quan trọng.
  
- [ ] **Sprint 2 — Baseline Retrieval + Answer (Lấy ngữ cảnh & Trả lời):**
  - *Code (`rag_answer.py`):* Truy vấn vector database ChromaDB bằng Dense Retrieval và đẩy ngữ cảnh cho LLM (OpenAI/Gemini) để trả lời.
  - *Yêu cầu kiến trúc (bắt buộc để nối Sprint 3):* Tách rõ 2 hàm `retrieve_context_baseline(query, k)` và `generate_answer(query, contexts)` để Sprint 3 chỉ thay logic retrieval, không đụng phần generation.
  - *Kiểm tra:* Kiểm tra test case. Câu hỏi có trong tài liệu phải có trích dẫn `[1]`, câu hỏi không có dữ liệu thực tế (như lỗi không có thực) hệ thống phải từ chối trả lời (abstain).

- [ ] **Sprint 3 — Tuning Tối Thiểu (Tối ưu hệ thống RAG):**
  - *Code (`rag_answer.py`):* Nâng cấp chất lượng truy xuất bằng cách **kế thừa retrieval của Sprint 2**. Chọn 1 trong 3 hướng đi: **Hybrid Search** (Sparse/BM25 + Dense), **Rerank** (Dùng cross-encoder đánh giá lại độ ưu tiên), hoặc **Query Transform** (Viết lại/Mở rộng câu hỏi query).
  - *Ràng buộc triển khai:* Tạo thêm hàm `retrieve_context_variant(query, k)` và giữ nguyên `generate_answer(...)` của Sprint 2. Luồng variant phải là: `query -> retrieve_context_variant -> generate_answer`.
  - *A/B bắt buộc:* Trong cùng `rag_answer.py`, có cờ `--mode baseline|variant` (hoặc biến cấu hình tương đương) để chạy cùng một pipeline với 2 retrieval khác nhau. Không tách thành 2 script độc lập.
  - *Tài liệu:* So sánh kết quả của bản mới nâng cấp (variant) với phiên bản gốc (baseline). Ghi chú những tinh chỉnh này vào `docs/tuning-log.md`.

- [ ] **Sprint 4 — Evaluation + Docs + Report (Đánh giá và Báo cáo):**
  - *Kiểm tra (`eval.py`):* Chạy script chấm điểm trên bộ 10 test questions có sẵn. Cần so sánh đo lường hiệu năng A/B test (baseline vs variant).
  - *Tài liệu:* Cập nhật cấu trúc hệ thống vào `docs/architecture.md`, và mọi người viết Báo Cáo Cá Nhân lưu trong `reports/individual/`.

---

## 3. Vai Trò Từng Thành Viên Nhóm

Tham gia nhóm, cần phân bổ nhiệm vụ rõ ràng:

| Vai trò | Phụ trách chính | Trách nhiệm chi tiết | Sprint Lead |
| --- | --- | --- | --- |
| **Tech Lead** | Code pipeline chạy end-to-end | Đảm bảo mã nguồn liên kết xuyên suốt các file. Chịu trách nhiệm hoàn thành khung ở Sprint 1, 2. | Sprint 1, 2 |
| **Retrieval Owner** | Thuật toán truy xuất (Retrieval) | Lên chiến lược phân tách chữ (chunking block, size, metadata). Ứng dụng Hybrid hay Rerank Tuning. | Sprint 1, 3 |
| **Eval Owner** | Đánh giá & QA (Evaluation) | Cấu hình test questions, expected evidence. Làm sổ số liệu (scorecard) thống kê các metric so sánh 2 mô hình. | Sprint 3, 4 |
| **Documentation Owner** | Lập tài liệu (Documentation) | Tổng hợp kiến trúc, sơ đồ(`architecture.md`), viết logic quá trình tuning, phân tích thay đổi biến số (`tuning-log.md`). | Sprint 4 |

> **Lưu ý yêu cầu cá nhân (40 Điểm):** Dù code chung nhưng **mỗi người đều phải nộp một báo cáo riêng chuyên sâu 500-800 từ (Individual Report)** phân tích rõ sự đóng góp, chọn phân tích nguyên nhân 1 lỗi trong 1 câu hỏi đánh giá thực tế và rút ra kinh nghiệm để lấy điểm cá nhân (40/100đ lab). 

---

## 3.1 Logic Liên Kết Giữa Sprint 1 -> Sprint 2 -> Sprint 3 (Bắt buộc)

- **Sprint 1 (Indexing) tạo dữ liệu nền:** sinh chunks + metadata và lưu ChromaDB. Nếu metadata sai hoặc chunks kém, Sprint 2 và Sprint 3 đều giảm chất lượng.
- **Sprint 2 (Baseline) tạo mốc chuẩn:** dùng `retrieve_context_baseline` + `generate_answer` để có chất lượng gốc làm chuẩn so sánh.
- **Sprint 3 (Variant) chỉ nâng retrieval:** mọi tối ưu phải đi qua `retrieve_context_variant`, còn bước generate giữ nguyên từ Sprint 2 để đảm bảo so sánh công bằng.
- **Nguyên tắc thực nghiệm:** chỉ đổi 1 biến chính ở Sprint 3 (Hybrid hoặc Rerank hoặc Query Transform), tránh đổi nhiều yếu tố cùng lúc gây khó kết luận.
- **Kết quả mong muốn:** `variant` cải thiện retrieval quality (precision/recall của context) và kéo theo chất lượng answer tốt hơn khi chạy `eval.py` ở Sprint 4.

> ✅ Checklist xác nhận đã "nối logic đúng": cùng nguồn index (Sprint 1), cùng generate function (Sprint 2), khác retrieval strategy (Sprint 3), và đo A/B trên cùng bộ câu hỏi.

---

## 4. Cách Thức Nộp Bài & Tổ Chức Repo

Đến giờ nộp bài (**18:00**), các nhóm phải đảm bảo Github Repository của mình có cấu trúc sau:

**Cấu trúc Repo phải nộp:**
```text
repo/
├── index.py                          # Code lập chỉ mục
├── rag_answer.py                     # Code Retrieve + suy luận câu trả lời LLM
├── eval.py                           # Code tự chạy test
├── data/
│   ├── docs/                         # Thư mục chứa 5 bộ document chính sách txt
│   └── test_questions.json           # Câu hỏi test để develop
├── logs/
│   └── grading_run.json              # File log chạy 10 câu hỏi GRADING ẩn (phát sau 17:00)
├── results/
│   ├── scorecard_baseline.md         # Kết quả chấm mẫu Base
│   └── scorecard_variant.md          # Kết quả chấm mẫu Tuning tùy chỉnh
├── docs/
│   ├── architecture.md               # Kiến trúc, cấu hình chunking hệ thống RAG pipeline
│   └── tuning-log.md                 # Giải thích lý do thiết kế variant ở Sprint 3
└── reports/
    ├── group_report.md               # Tổng kết nhóm chung (cho nhóm 3 người)
    └── individual/
        └── [ten_thanh_vien].md       # Từng người 1 file báo cáo cá nhân
```

**Timeline nộp bài nghiêm ngặt:**
- **17:00:** Mở file `grading_questions.json` (10 bài test quyết định điểm nhóm).
- **17:00 - 18:00:** Chạy pipeline để xuất file `logs/grading_run.json`.
- **18:00 (Deadline Code):** Commit toàn bộ code file `.py`, files tĩnh (`.json`) và tài liệu nhóm. **Hệ thống khoá code**.
- **Sau 18:00:** Chỉ cho phép commit báo cáo Markdown (`reports/group_report.md` và `reports/individual/[ten].md`). Mọi sửa chữa Code đều bị vô hiệu hoá điểm.

🚨 **Hậu Quả Quy Phạt (Trừ thẳng vào điểm 0):** Cả nhóm sẽ bị **-50% số điểm của câu hỏi** đó nếu cố tình bịa thông tin thay vì báo lỗi/abstain. **Mất toàn bộ điểm cá nhân (0/40)** nếu: nhận công của người khác nhưng lúc tra khảo không chứng minh được, report không đúng với lịch sử commit, report copy-paste người khác.

---

## 5. Các Tính Năng Làm Thêm (Bonus) Lấy Điểm Thưởng (Tối đa +5)
Nếu muốn đột phá và sở hữu số điểm tuyệt đối, hãy thực hiện các cải tiến công nghệ sau để hệ thống vượt trội:

1. **Áp dụng AI LLM-as-Judge tự chấm bài (+2 Điểm):** Trong file `eval.py`, thay vì kiểm tra bằng rule/check tay, hãy tích hợp LLM làm "Giám khảo" phân tích độ bao phủ bằng Prompt để đánh giá và chấm điểm Faithfulness/Relevance tự động.
2. **Kỷ luật Logging (+1 Điểm):** Đảm bảo log đầu ra tại file `logs/grading_run.json` không thiếu câu nào (10/10) và Timestamp in ra ở các file phải diễn ra trong khung thời gian hẹp từ 17:00 đến 18:00.
3. **Thách Thức Câu Hỏi Multi-Hop `gq06` (+2 Điểm):** Chinh phục câu hỏi ẩn mức khó cao nhất (gq06) mang về 12 điểm Raw + 2 điểm Bonus nếu đúng hoàn toàn (Full marks). Nó đòi hỏi pipeline RAG phải lùng quét thông tin xuyên suốt nhiều tài liệu (multi-document synthesis) và reasoning vượt trội.
