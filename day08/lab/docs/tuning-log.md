# Tuning Log — RAG Pipeline (Day 08 Lab)

> A/B Rule: Chỉ đổi **MỘT biến** mỗi lần để biết rõ tác động.

---

## Baseline (Sprint 2)

**Ngày:** 2026-04-13  
**Config:**
```text
retrieval_mode = "dense"
chunk_size = 400 tokens
overlap = 80 tokens
top_k_search = 10
top_k_select = 3
use_rerank = False
llm_model = gpt-4o-mini
```

**Scorecard Baseline (manual scoring):**
| Metric | Average Score |
|--------|--------------|
| Faithfulness | 2.90 /5 |
| Answer Relevance | 4.10 /5 |
| Context Recall | 5.00 /5 |
| Completeness | 3.30 /5 |

**Câu hỏi yếu nhất (điểm thấp):**
- `q09` (ERR-403-AUTH): relevance và completeness thấp do đây là câu thiếu context thực, model trả abstain ngắn.
- `q04` (refund digital product): completeness thấp vì answer chưa bao phủ đủ phần ngoại lệ.
- `q07` (approval matrix alias): có lúc baseline tốt hơn variant ở faithfulness, cho thấy answer quality còn phụ thuộc generation.

**Giả thuyết nguyên nhân (Error Tree):**
- [x] Retrieval: Dense có thể hụt alias/keyword đặc thù.
- [x] Generation: Có câu trả lời đúng hướng nhưng thiếu key points.
- [ ] Indexing metadata thiếu (không phải vấn đề chính trong run này).

---

## Variant 1 (Sprint 3)

**Ngày:** 2026-04-13  
**Biến thay đổi:** `retrieval_mode` từ `dense` -> `hybrid`  
**Lý do chọn biến này:**
Corpus có cả câu tự nhiên (policy/HR) và keyword đặc thù (SLA P1, ERR-403, Level 3).  
Hybrid kết hợp dense + sparse để tăng khả năng bắt đúng chunk chứa alias/mã lỗi mà vẫn giữ semantic matching.

**Config thay đổi:**
```text
retrieval_mode = "hybrid"
top_k_search = 10
top_k_select = 3
use_rerank = False
```

> Lưu ý: giữ nguyên top-k và generation để tuân thủ A/B rule (chỉ đổi 1 biến retrieval).

### Kết quả A/B — Manual
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 2.90/5 | 2.90/5 | +0.00 |
| Answer Relevance | 4.10/5 | 4.20/5 | +0.10 |
| Context Recall | 5.00/5 | 5.00/5 | +0.00 |
| Completeness | 3.30/5 | 3.40/5 | +0.10 |

### Kết quả A/B — LLM-as-Judge
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 3.50/5 | 3.80/5 | +0.30 |
| Answer Relevance | 4.30/5 | 4.40/5 | +0.10 |
| Context Recall | 5.00/5 | 5.00/5 | +0.00 |
| Completeness | 3.50/5 | 3.80/5 | +0.30 |

**Nhận xét theo câu hỏi (manual run):**
- Variant tốt hơn ở `q06`, `q08` (multi-detail/cross-context) do hybrid giữ được cả semantic lẫn keyword.
- Baseline tốt hơn ở `q07` về faithfulness trong run này (khả năng do generation chọn diễn đạt khác dù context tương đương).
- Nhiều câu tie, cho thấy retrieval đã ổn định nhưng generation vẫn là bottleneck ở một số câu.

**Kết luận Variant 1:**
Variant hybrid **tốt hơn nhẹ** so với baseline theo relevance/completeness và ổn định hơn trong đánh giá LLM-as-Judge.  
Vì vậy nhóm chọn `variant_hybrid` làm cấu hình chính cho grading run.

---

## Variant 2

Không thực hiện để tránh vi phạm A/B rule trong khung thời gian lab.

---

## Tóm tắt học được

1. **Lỗi phổ biến nhất trong pipeline này là gì?**  
   Khác biệt giữa retrieval đúng và answer đầy đủ: nhiều câu retrieve đúng source nhưng answer vẫn thiếu key points.

2. **Biến nào có tác động lớn nhất tới chất lượng?**  
   `retrieval_mode` (dense -> hybrid) tác động rõ nhất ở câu chứa alias/keyword và câu multi-detail.

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**  
   Bật rerank cross-encoder trong variant thứ 2 để giảm noise ở top-k và cải thiện faithfulness/completeness.

---

## File bằng chứng

- Baseline manual: `results/scorecard_baseline.md`
- Variant manual: `results/scorecard_variant.md`
- A/B manual: `results/ab_comparison.csv`
- Baseline judge: `results/scorecard_baseline_judge.md`
- Variant judge: `results/scorecard_variant_judge.md`
- Grading run chính: `logs/grading_run.json`
- Grading run judge (so sánh): `logs/grading_run_judge.json`
