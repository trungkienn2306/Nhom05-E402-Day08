# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Bùi Thế Công  
**Vai trò trong nhóm:** Tech Lead / Documentation Owner (chịu trách nhiệm sprint 2; chạy test và review docs trong sprint 4)
**Ngày nộp:** 13/04/2026  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong Lab Day 08, em chủ yếu phụ trách **Sprint 2** (Baseline Retrieval + Generation) và tham gia **Sprint 4** (Evaluation) cùng cả nhóm.

Ở Sprint 2, em implement ba thành phần cốt lõi trong `rag_answer.py`: (1) hàm `retrieve_dense()` — truy vấn ChromaDB bằng embedding cosine similarity, kế thừa `get_embedding()` từ Sprint 1; (2) hàm `call_llm()` — gọi OpenAI GPT-4o-mini với `temperature=0` để đảm bảo output ổn định khi evaluate; (3) hàm `build_grounded_prompt()` — xây dựng prompt ép model chỉ trả lời từ context được retrieve, có citation dạng `[1]`, và abstain nếu thiếu dữ liệu. Sau đó, em chạy test rag_answer() với nhiều câu hỏi mẫu để đảm bảo các câu trả lời của LLM luôn dựa trên context được retrieve và trích dẫn đúng nguồn tồn tại; nếu không đủ thông tin => LLM trả về "Tôi không biết". 

Ở Sprint 4, nhóm cùng chạy `eval.py` để lấy scorecard baseline và variant, sau đó so sánh A/B bằng `compare_ab()`. Cuối cùng, điền và review kết quả trong 2 docs architecture.md và tuning-log.md.  

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau lab này, em hiểu rõ hơn hai điều:

**Thứ nhất: Grounded Prompt là "hàng rào" chống hallucination, không phải viết thêm vào cho đẹp.** Trước khi làm lab, em nghĩ prompt đơn giản là hỏi LLM trả lời. Thực tế, nếu không có câu ràng buộc `"Answer only from the retrieved context"` kết hợp với `"if the context is insufficient, say you do not know"`, model sẽ tự điền từ kiến thức chung của nó — và rất tự tin khi làm vậy. Với câu hỏi `gq07` (phạt SLA), model không có context nên phải abstain — đây là hành vi đúng mà prompt phải ép được.

**Thứ hai: Evaluation không phải để chấm điểm cuối — mà là để biết lỗi ở đâu trong pipeline.** Khi thấy `q01` (SLA P1) có Faithfulness thấp (2/5) dù Context Recall = 5/5, em nhận ra: vấn đề không phải retrieval — mà là generation không bám context đủ chặt. Đây là phân biệt quan trọng mà chỉ có scorecard mới lộ ra được.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

**Khó khăn lớn nhất:** Khó nhất với em là **tinh chỉnh config để đạt điểm tốt nhất một cách ổn định**, không phải chỉ làm cho pipeline chạy được. Nhóm phải thử nhiều tổ hợp như `top_k_search`, `top_k_select`, `retrieval_mode` (dense/hybrid) và bật/tắt rerank, nhưng vẫn phải giữ A/B rule (mỗi lần chỉ đổi một biến) để biết chính xác biến nào tạo cải thiện. Có những lần Context Recall tăng nhưng Faithfulness không tăng tương ứng, nên phải đọc kỹ từng câu trong scorecard để xác định bottleneck nằm ở retrieval hay generation. Phần này tốn thời gian nhất vì cần vừa chạy nhiều vòng test vừa phân tích nguyên nhân sai lệch theo từng metric.

**Điều ngạc nhiên nhất:** Em tưởng `temperature=0` sẽ khiến GPT-4o-mini trả lời rất máy móc, nhưng thực tế model vẫn suy luận linh hoạt và viết câu tự nhiên. Sự khác biệt là output ổn định — chạy cùng một câu hỏi nhiều lần cho kết quả gần như giống nhau, rất quan trọng khi evaluate.

**Kết quả bất ngờ:** Khi chạy scorecard baseline, Context Recall đạt 5/5 cho hầu hết câu hỏi, nhưng Faithfulness chỉ 2.90/5. Em ngỡ rằng retrieve đúng thì answer phải tốt — thực ra retrieve đúng chỉ là điều kiện cần, generation mới quyết định câu trả lời có grounded không.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** `gq07` — *"Công ty sẽ phạt bao nhiêu nếu team IT vi phạm cam kết SLA P1?"*

**Phân tích:**

Đây là câu hỏi thuộc loại **"hallucination bait"** — câu hỏi có vẻ hợp lý nhưng tài liệu nội bộ không có thông tin về mức phạt. Pipeline đúng phải abstain.

**Baseline (dense):** Scorecard ghi Faithfulness = 1/5, Relevance = 1/5, Completeness = 1/5. Nghe qua tưởng pipeline thất bại — nhưng thực ra đây là kết quả tốt về mặt behavior. Model trả lời rằng không có thông tin trong tài liệu hiện có (abstain đúng). Điểm thấp là do cách tính thủ công không nhận ra abstain cho câu "insufficient context" là đúng (expected_sources = []).

**Lỗi ở đâu:** Không phải lỗi của pipeline — lỗi nằm ở **scoring rule**: hàm `score_faithfulness()` cho điểm 1 khi abstain mà không kiểm tra xem câu hỏi đó có expected_sources hay không. Khi bật `USE_LLM_JUDGE=1`, LLM judge hiểu ngữ cảnh hơn và cho điểm cao hơn cho hành vi abstain đúng.

**Variant (hybrid):** Kết quả tương tự baseline — vì lỗi không nằm ở retrieval mà ở evaluation logic. Đây là bài học: cải thiện retrieval không giúp ích nếu bottleneck là generation/evaluation.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, em sẽ làm hai việc. Thứ nhất, thử xây một **agent RAG** trong đó retriever được đóng gói như một tool để agent có thể chủ động gọi truy xuất theo từng bước reasoning. Thứ hai, tự động hóa CI/CD bằng cách tích hợp vòng lặp **RAGAS evaluation vào GitHub Actions** (chạy baseline/variant, lưu scorecard artifact, cảnh báo khi quality giảm) để quá trình tuning có thể lặp lại và kiểm soát chất lượng liên tục.

---

*Lưu file này với tên: `reports/individual/bui_the_cong.md`*
