# Tuning Log — RAG Pipeline (Day 08 Lab)

> Template: Ghi lại mỗi thay đổi và kết quả quan sát được.
> A/B Rule: Chỉ đổi MỘT biến mỗi lần.

---

## Baseline (Sprint 2)

**Ngày:** ___________  
**Config:**
```
retrieval_mode = "dense"
chunk_size = _____ tokens
overlap = _____ tokens
top_k_search = 10
top_k_select = 3
use_rerank = False
llm_model = _____
```

**Scorecard Baseline:**
| Metric | Average Score |
|--------|--------------|
| Faithfulness | ? /5 |
| Answer Relevance | ? /5 |
| Context Recall | ? /5 |
| Completeness | ? /5 |

**Câu hỏi yếu nhất (điểm thấp):**
> TODO: Liệt kê 2-3 câu hỏi có điểm thấp nhất và lý do tại sao.
> Ví dụ: "q07 (Approval Matrix) - context recall = 1/5 vì dense bỏ lỡ alias."

**Giả thuyết nguyên nhân (Error Tree):**
- [ ] Indexing: Chunking cắt giữa điều khoản
- [ ] Indexing: Metadata thiếu effective_date
- [ ] Retrieval: Dense bỏ lỡ exact keyword / alias
- [ ] Retrieval: Top-k quá ít → thiếu evidence
- [ ] Generation: Prompt không đủ grounding
- [ ] Generation: Context quá dài → lost in the middle

---

## Variant 1 (Sprint 3)

**Ngày:** ___________  
**Biến thay đổi:** ___________  
**Lý do chọn biến này:**
> TODO: Giải thích theo evidence từ baseline results.
> Ví dụ: "Chọn hybrid vì q07 (alias query) và q09 (mã lỗi ERR-403) đều thất bại với dense.
> Corpus có cả ngôn ngữ tự nhiên (policy) lẫn tên riêng/mã lỗi (ticket code, SLA label)."

**Config thay đổi:**
```
retrieval_mode = "hybrid"   # hoặc biến khác
# Các tham số còn lại giữ nguyên như baseline
```

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | ?/5 | ?/5 | +/- |
| Answer Relevance | ?/5 | ?/5 | +/- |
| Context Recall | ?/5 | ?/5 | +/- |
| Completeness | ?/5 | ?/5 | +/- |

**Nhận xét:**
> TODO: Variant 1 cải thiện ở câu nào? Tại sao?
> Có câu nào kém hơn không? Tại sao?

**Kết luận:**
> TODO: Variant 1 có tốt hơn baseline không?
> Bằng chứng là gì? (điểm số, câu hỏi cụ thể)

---

## Variant 2 (nếu có thời gian)

**Biến thay đổi:** ___________  
**Config:**
```
# TODO
```

**Scorecard Variant 2:**
| Metric | Baseline | Variant 1 | Variant 2 | Best |
|--------|----------|-----------|-----------|------|
| Faithfulness | ? | ? | ? | ? |
| Answer Relevance | ? | ? | ? | ? |
| Context Recall | ? | ? | ? | ? |
| Completeness | ? | ? | ? | ? |

---

## Tóm tắt học được

> TODO (Sprint 4): Điền sau khi hoàn thành evaluation.

1. **Lỗi phổ biến nhất trong pipeline này là gì?**
   > _____________

2. **Biến nào có tác động lớn nhất tới chất lượng?**
   > _____________

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
   > _____________


---

## [AUTO] Experiment Log — 2026-04-13 15:23

**Biến thay đổi:** `retrieval_mode`: `dense` → `hybrid` + `use_rerank`: `False` → `True`

**Lý do chọn biến này:**
> Corpus IT Helpdesk có 2 loại query:
> 1. Câu ngôn ngữ tự nhiên ("Chính sách hoàn tiền như thế nào?") → Dense mạnh
> 2. Keyword chính xác ("ERR-403", "P1 ticket", "Level 3 access") → BM25 mạnh
> Hybrid RRF kết hợp cả 2 để không bỏ sót câu hỏi dạng nào.
> CrossEncoder rerank loại bỏ chunk noise sau khi search rộng.

**Baseline Config:** `baseline_dense`
```
retrieval_mode = "dense"
top_k_search   = 10
top_k_select   = 3
use_rerank     = False
```

**Variant Config:** `variant_hybrid_rerank`
```
retrieval_mode = "hybrid"
top_k_search   = 10
top_k_select   = 3
use_rerank     = True
reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

**Tổng câu hỏi chạy:** 1

**Quan sát cải thiện (Variant > Baseline):**
- [q01] Variant retrieve được 4 chunks vs 3 (baseline)

**Quan sát hồi quy (Baseline > Variant):**
- Không có

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

**Notes:** [DRY RUN TEST] Không phải kết quả thực tế
