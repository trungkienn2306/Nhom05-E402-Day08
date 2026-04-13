# Group Report — Technical Decision Log (Day 08 RAG Lab)

**Team:** Nhom05-E402-Day08  
**Date:** 13/04/2026  
**Scope:** Tập trung vào quyết định kỹ thuật cấp nhóm.

---

## 1) Problem framing

Nhóm xây pipeline RAG cho dữ liệu nội bộ CS + IT Helpdesk với yêu cầu:
- Trả lời dựa trên evidence (có citation).
- Không bịa khi thiếu ngữ cảnh (abstain).
- Đo được chất lượng baseline vs variant theo A/B.

Ràng buộc thực thi:
- Corpus nhỏ (5 tài liệu) nhưng đa dạng kiểu query (natural language + keyword/mã lỗi).
- Do thời gian ngắn theo sprint (mỗi sprint 60 phút), cần ưu tiên quyết định có tác động lớn nhất.

---

## 2) Decision log cấp nhóm

## D1 — Chọn chiến lược chunking 2 tầng (heading-first + size fallback)
**Quyết định:**  
Dùng tách theo heading `=== ... ===` trước, sau đó mới chia theo kích thước có overlap.

**Vì sao chọn:**  
Nếu cắt cứng ngay từ đầu, điều khoản dễ bị tách giữa câu và làm mất ngữ nghĩa retrieval.

**Trade-off:**  
- Ưu điểm: giữ ngữ cảnh tốt hơn, tăng khả năng grounded answer.  
- Nhược điểm: số chunk không đồng đều, một số chunk dài hơn mức lý tưởng.

**Kết luận:**  
Giữ nguyên cho baseline và variant để ổn định A/B.

---

## D2 — Chuẩn metadata bắt buộc cho mọi chunk
**Quyết định:**  
Mỗi chunk bắt buộc có `source`, `section`, `effective_date`.

**Vì sao chọn:**  
Metadata không chỉ phục vụ citation mà còn là nền cho filter/routing theo domain ở các vòng cải tiến.

**Trade-off:**  
Tăng công preprocess, nhưng giảm rủi ro debug mù khi retrieval sai.

**Kết luận:**  
Metadata được coi là contract dữ liệu giữa indexing và retrieval.

---

## D3 — Baseline retrieval dùng dense-only
**Quyết định:**  
Baseline của nhóm cố định là dense retrieval (`retrieval_mode="dense"`), `top_k_search=10`, `top_k_select=3`, không rerank.

**Vì sao chọn:**  
Cần một mốc tham chiếu đơn giản, ổn định, dễ lặp lại để làm đối chứng cho tuning.

**Trade-off:**  
- Ưu điểm: đơn giản, ít phụ thuộc, dễ vận hành.  
- Nhược điểm: hụt query keyword cứng hoặc alias (ví dụ dạng mã lỗi).

**Kết luận:**  
Baseline đủ tốt để chạy toàn bộ eval và làm chuẩn so sánh.

---

## D4 — Grounded prompting + abstain policy
**Quyết định:**  
Ép generation theo nguyên tắc evidence-only, có citation, và trả lời "không biết/không đủ dữ liệu" khi context thiếu.

**Vì sao chọn:**  
Với hệ thống nội bộ, sai nhưng tự tin nguy hiểm hơn không trả lời.

**Trade-off:**  
Tăng tỷ lệ abstain ở câu biên, đổi lại giảm hallucination.

**Kết luận:**  
Chấp nhận conservative behavior để ưu tiên độ tin cậy.

---

## D5 — Chọn hướng tuning chính: Hybrid Retrieval (Dense + BM25)
**Quyết định:**  
Variant chính thức ưu tiên Hybrid bằng Reciprocal Rank Fusion (RRF), tùy chọn rerank là bước phụ.

**Vì sao chọn:**  
Corpus và query có cả ngôn ngữ tự nhiên lẫn keyword/mã lỗi; dense và sparse bổ sung điểm mạnh cho nhau.

**Trade-off:**  
- Ưu: tăng recall cho query dạng keyword.  
- Nhược: có thể kéo thêm chunk nhiễu, cần kiểm soát ở bước select/rerank.

**Kết luận:**  
Hybrid là lựa chọn hợp lý nhất cho dữ liệu hỗn hợp của nhóm.

---

## D6 — Áp dụng A/B rule nghiêm ngặt
**Quyết định:**  
Mỗi run chỉ thay một biến chính; không trộn đồng thời nhiều thay đổi (chunking + retrieval + rerank + prompt).

**Vì sao chọn:**  
Nếu đổi nhiều biến cùng lúc sẽ không xác định được nguyên nhân cải thiện hay suy giảm.

**Trade-off:**  
Tốn thêm thời gian chạy nhiều vòng nhưng kết luận có giá trị kỹ thuật cao hơn.

**Kết luận:**  
A/B rule là nguyên tắc quản trị thí nghiệm bắt buộc của nhóm.

---

## D7 — Nâng evaluator từ parse cứng sang parser robust + LLM-as-Judge định hướng
**Quyết định:**  
Cải tiến parser output judge để giảm lỗi parse; định hướng dùng LLM-as-Judge cho các metric ngữ nghĩa.

**Vì sao chọn:**  
Rule-based scoring kiểu token overlap dễ đánh sai các câu paraphrase đúng nghĩa hoặc case abstain đúng.

**Trade-off:**  
- Ưu điểm: phản ánh chất lượng thực hơn ở level ngữ nghĩa.  
- Nhược điểm: tăng chi phí và phụ thuộc LLM trong vòng eval.

**Kết luận:**  
Giữ scorecard hiện tại để đối chiếu, đồng thời coi LLM-as-Judge là hướng chuẩn hóa tiếp theo.

---

## 3) Những gì nhóm cố ý KHÔNG làm (de-scoping)

- Không thay đổi chunking trong cùng run với hybrid để tránh nhiễu A/B.
- Không tối ưu prompt quá mức khi chưa ổn định retrieval/evaluation.
- Không thêm nhiều variant đồng thời trong Sprint 3 vì rủi ro thiếu thời gian verify.

---

## 4) Kết quả kỹ thuật cấp nhóm

- Pipeline end-to-end đã vận hành: index -> retrieve -> generate -> eval.
- Có baseline và variant để so sánh.
- Có dữ liệu kết quả trong `results/` và log quyết định trong `docs/tuning-log.md`.
- Đã xác định rõ bottleneck trọng tâm: không chỉ retrieval mà còn nằm ở quality của evaluator và generation discipline.

---

## 5) Next technical decisions (nếu tiếp tục iteration)

1. Chuẩn hóa abstain text về một format duy nhất để pass rubric ổn định.  
2. Bổ sung metadata pre-filter theo domain (`department`) trước retrieval.  
3. Chạy matrix thí nghiệm tách riêng: `dense`, `hybrid`, `hybrid+rerank`.  
4. Đưa RAGAS/LLM-as-Judge vào CI (GitHub Actions) để canh regression tự động.

---

## 6) Kết luận

Giá trị chính của nhóm nằm ở việc ra quyết định kỹ thuật có kiểm soát:  
thiết kế baseline rõ ràng, tuning theo giả thuyết, đo bằng A/B, và tách bạch được nguyên nhân lỗi theo từng tầng của RAG pipeline.

