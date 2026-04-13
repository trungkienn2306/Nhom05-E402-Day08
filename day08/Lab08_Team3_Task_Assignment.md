# Quy Hoạch Công Việc Lab Day 08 Cân Bằng Cho Nhóm 3 Người (Mỗi Người 1 Sprint)

Dựa trên yêu cầu mỗi thành viên làm chủ một Sprint riêng biệt để tập trung chuyên môn sâu, kết hợp 4 Sprint của đề bài cho 3 thành viên, chúng ta sẽ có cơ cấu nhóm như sau: 3 thành viên chia nhau 3 Sprint kỹ thuật cốt lõi. Riêng Sprint 4 (Đánh giá và Báo cáo) - đây là sprint văn bản và tổng hợp, nên sẽ được bóc tách và chia đều lại cho cả 3.

---

## 1. Phân Chia Vai Trò & Sprint Cốt Lõi
*(Tham khảo và điền tên thành viên vào cột Thành viên)*

| Vai trò | Thành viên phụ trách | Nhiệm vụ kỹ thuật chính (Code) | Nhiệm vụ Đánh giá/Tài liệu (Thuộc Sprint 4) |
| --- | --- | --- | --- |
| **Dev 1: Tech Lead** | `[Tên TV 1]` | **Làm chủ Sprint 1:** Build Index & Chunking | Viết file `docs/architecture.md` (Tổng hợp sơ đồ) |
| **Dev 2: Data & RAG Owner** | `[Tên TV 2]` | **Làm chủ Sprint 2:** Baseline Retrieval + Answer | Setup bộ `data/test_questions.json` và hỗ trợ Dev 3 chạy Log grading |
| **Dev 3: Tuning & QA Owner** | `[Tên TV 3]` | **Làm chủ Sprint 3:** Tuning Variante (Rerank/Hybrid) | Code bộ `eval.py` và viết lịch sử `tuning-log.md` |

> **Lưu ý Bắt Buộc:** Mỗi cá nhân (kể cả Dev 1, 2, 3) đều TỰ MÌNH phải viết 1 bài **"Báo cáo cá nhân"** (500-800 từ vào `reports/individual/[tên].md`). Để đạt 40 điểm cá nhân, trong báo cáo phải bóc tách 1 câu hỏi test grading thực tế và chứng minh tự phân tích lỗi. Cấm sao chép!

---

## 2. Chi Tiết Công Việc Từng Thành Viên Theo Từng Phút

### 👤 Dev 1 (Làm chủ Sprint 1: Build Index)
*Người đặt nền móng dữ liệu (Khâu Data Ingestion)*
- **Nhiệm vụ Code (`index.py`):** Lập trình logic nạp, cắt nhỏ dữ liệu (Chunking) từ 5 tài liệu hợp đồng có sẵn. Thiết lập OpenAI API key / Sentence Transformers tạo Embeddings và đưa dữ liệu lên ChromaDB.
- **Tiêu chí hoàn thành (DoD):** Dữ liệu phải có đầy đủ 3 metadata: `source`, `section`, `effective_date`.
- **Nghĩa vụ Sprint 4:** Vẽ/Viết file `architecture.md`. Nêu bật cấu hình Chunking bạn vừa làm (size, overlap, why?) và sơ đồ kỹ thuật Pipeline của team.
- **Tính năng Bonus:** Setup Github Repo đầu giờ cho nhóm clone về.

### 👤 Dev 2 (Làm chủ Sprint 2: Baseline Retrieval)
*Người gánh luồng code chính để hệ thống biết "Trả lời"*
- **Nhiệm vụ Code (`rag_answer.py`):** Viết logic hàm `retrieve_dense()` lấy dữ liệu thông minh trong DB do Dev 1 vừa đẩy lên. Nối với API của Claude/GPT-4 để khai báo Prompt.
- **Tiêu chí hoàn thành (DoD):** Model trả lời được thông tin SLA lấy kèm nguồn trích dẫn là `[1]`. Cài đặt được việc "Từ chối khéo (Abstain)" để giảm mức Hallucinate về 0 khi bị hỏi đểu thông tin không nằm trong tài liệu.
- **Nghĩa vụ Sprint 4:** Tự nghĩ ra/cập nhật bộ 10 câu hỏi để test tính năng trong `test_questions.json`.
- **Tính năng Bonus:** Đảm bảo điểm số cao cho câu hỏi ẩn đa suy luận khó nhất (`gq06`) xuất hiện lúc 17:00.

### 👤 Dev 3 (Làm chủ Sprint 3: Tuning & Lãnh đạo Sprint 4)
*Người đánh giá và tối ưu trải nghiệm cuối cùng*
- **Nhiệm vụ Code (`rag_answer.py` & `eval.py`):** 
  - Code Sprint 3: Tạo ra 1 bản phái sinh Variant mạnh mẽ hơn bản Baseline. Chọn triển khai Hybrid Search (gộp từ khoá) hoặc Reranking (hệ thống xếp hạng chéo).
  - Code Sprint 4: Nạp bộ so sánh tự động (Scorecard) để chạy A/B Test so đo điểm 2 hệ.
- **Tiêu chí hoàn thành (DoD):** Hệ thống Variant chạy end-to-end mượt. Bảng scorecard output cho thấy lý do chọn thiết kế đó.
- **Nghĩa vụ Sprint 4:** Viết biên bản công chiếu thử nghiệm - `docs/tuning-log.md`. Viết báo cáo tổng hợp kết quả của cả đội (`group_report.md`).
- **Tính năng Bonus:** Triển khai cơ chế LLM-as-a-Judge trong file chấm bài thay cho rule thủ công (+2Đ).

---

## 3. Timeline Đề Xuất Của Team

| Thời gian thực do | Mục tiêu chung cả team cần chốt sổ |
| --- | --- |
| **Giờ thứ 1** | **Dev 1** code xong Sprint 1. CSDL (ChromaDB) được hình thành. Dev 2 và Dev 3 đọc tài liệu phân tích logic cấu trúc dữ liệu. |
| **Giờ thứ 2** | **Dev 2** code xong Sprint 2. Hệ thống bắt đầu test hỏi đáp mượt mà, trả ra citations. |
| **Giờ thứ 3** | **Dev 3** code xong Sprint 3. Chạy được hệ thống RAG mới (Variant) mạnh hơn, song song Dev 1 bắt đầu viết docs gửi trước. |
| **Giờ thứ 4** | **Cả nhóm vào Sprint 4.** Dev 3 chạy code file chấm bài A/B Eval. Dev 1 và Dev 2 phụ chốt Docs `architecture` & `tuning_log`. |
| **17:00 - 18:00**| **Giờ G:** Lấy file `grading_questions.json` test mù và chạy lệnh sinh file log chấm thi. Hỗ trợ nhau diệt bug Crash (+1Đ bonus log). |
| **Đúng 18:00** | **Khoá Code** (Push chốt sổ `*.py`, `logs`, `results`, `architecture.md`, `tuning-log.md`). |
| **Tối (Sau 18:00)**| Mở thong thả file Markdown. Gắn text các bài **báo cáo cá nhân của 3 người**, báo cáo nhóm và đẩy commit muộn. |
