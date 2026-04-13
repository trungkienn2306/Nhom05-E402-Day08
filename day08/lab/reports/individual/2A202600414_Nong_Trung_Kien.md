# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Nông Trung Kiên  
**Vai trò trong nhóm:** Dev 3 (Tuning & QA Owner)  
**Ngày nộp:** 13/04/2026  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Em đã làm gì trong lab này? (100-150 từ)

Trong dự án lab này, em đảm nhận trọng trách và phụ trách chính ở Sprint 3 và Sprint 4. 

Ở Sprint 3, em đã thực hiện tinh chỉnh và phát triển module tìm kiếm nâng cao. Cụ thể, em thiết kế thuật toán Tìm kiếm lai (Hybrid Search) bằng việc kết hợp truy xuất từ khóa (Sparse/BM25) và truy xuất ngữ nghĩa (Dense Embedding). Em trực tiếp sửa đổi và tối ưu hàm `retrieve_context_variant` để cải thiện chất lượng thu thập bối cảnh văn bản.

Bước sang Sprint 4, em đã lập trình module kiểm định (Evaluation) cấu hình tự động tính toán bộ điểm số (Scorecard) trong tệp lệnh `eval.py`. Em đã ứng dụng thiết kế chấm chéo hiện đại LLM-as-a-Judge trong phần tính điểm để đánh giá tự động Độ trung thực (Faithfulness) và Độ liên quan (Answer Relevance) thay vì tự kiểm định thủ công mất thời gian. Công việc tinh chỉnh của em tích hợp thẳng vào luồng đầu ra của bộ nhúng từ Dev 1 ở Sprint 1 và sử dụng hàm Sinh ngôn (Generation) nguyên bản của Dev 2 ở Sprint 2.

---

## 2. Điều em hiểu rõ hơn sau lab này (100-150 từ)

Khái niệm đầu tiên em thấu hiểu tường tận chính là **Hybrid Retrieval**. Trước đây, em thường thiên vị áp đặt rằng Dense Retrieval (Truy xuất nhúng) sẽ xử lý được tất cả vì độ hiểu ngữ nghĩa (semantic understanding). Nhưng khi kiểm duyệt dữ liệu có từ khóa chặt hẹp như mã lỗi phần mềm (ERR-403) hay các từ khóa riêng biệt (SLA P1), Dense Retrieval lộ rõ khiếm khuyết. Hybrid Search bù đắp lỗ hổng này hoàn hảo; Sparse tìm khớp chuỗi chính xác trong khi Dense bao trọn ngữ nghĩa, qua đó triệt tiêu điểm mù đặc thù của không gian vector.

Kế đó, **LLM-as-a-Judge** trở thành kỹ thuật thú vị thứ hai em lĩnh hội được. Việc biến chính một LLM (như mô hình ngôn ngữ lớn) trở thành "vị giám khảo" soi chiếu kết quả pipeline RAG giúp em kiểm soát khách quan hơn với độ bao phủ dữ liệu (Completeness) hay tính trung thực (Faithfulness). Em học được cách "ép" LLM không sinh sinh chuỗi tự do (Generation) mà trở thành một máy đo qua việc sử dụng thiết kế cấu trúc chỉ dẫn đánh giá (Prompting Evaluation) chặt chẽ.

---

## 3. Điều em ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Điều gây bất ngờ nhất cũng là nút thắt hao bớt nhiều công sức (debug) nhất tới từ sự nhập nhằng giữa **lỗi Truy xuất (Retrieval) và Lỗi Ngôn ngữ (Generation)**.

Ban đầu em cứ mặc định lỗi Độ bao phủ (Completeness) thấp là do Truy xuất (Retrieval) đã trích xuất sót dữ liệu. Nhưng sự thật không tương ứng 100%. Khi dò xuất tệp nhật ký nhúng (Log), em ngạc nhiên phát hiện khối (Chunk) chính xác đã được mô hình đưa trả vào ngữ cảnh (Context), tuy nhiên, lớp mô hình sinh văn (LLM Generation) của Sprint 2 lại ngó lơ thông tin rườm rà và trả lời rút gọn, khiến hệ thống giám định đánh giá thiếu ý. 

Việc theo dõi (Trace) một biến số gây lỗi đòi hỏi em phải chia việc đánh giá (eval loop) ra nhiều khâu, kiểm tra từng log của `retrieve_context_variant()`. Cách biệt lập từng cơ chế mới chứng thực được lỗi xuất phát tại khâu nào, đây cũng là thao tác nhọc nhằn nhất nhưng quan trọng nhất trong cả ca lab.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** `q09` - Câu hỏi yêu cầu xử lý mã lỗi bảo mật "ERR-403-AUTH".

**Phân tích:**
Trong lần đợt quét đầu tiên bằng cấu hình chuẩn (Baseline), mô hình tỏ ra lúng túng, trả lời thiếu các bước kiểm định kỹ thuật hoặc đôi khi tự từ chối trả lời (Abstain). Điểm ghi nhận thủ công cho hạng mục Độ liên quan (Relevance) và Độ bao phủ (Completeness) suy giảm.

Sau khi trực tiếp giải phẫu (Debug), em phát hiện lỗi gốc phát tác ngay từ khâu **Indexing/Retrieval**. Cụ thể, Dense Retrieval thông qua Embedding lại ưu tiên nhóm nghĩa tương đồng ngữ nghĩa hơn là độ khớp từng kí tự. Cụm Alias cực kì đặc biệt như `ERR-403-AUTH` bị vector đánh trượt nên rớt khỏi tệp danh sách Top K Bối cảnh (Context blocks). Mô hình ngôn ngữ hệ quả sinh ra (Generation) đương nhiên gặp ảo giác (Hallucination) hoặc Abstain vì mất gốc.

Khi nâng cấp cấu hình Nâng cao (Variant) qua Hybrid Search, hệ số đo lường BM25 (Sparse) lập tức tìm khớp cực nhạy được nhãn hiệu trên và triệu hồi đoạn Chunk chính xác. Nguồn tin bơm vào đủ giúp bản Variant bật tăng điểm `Answer Relevance` trung bình hệ thống từ 4.30/5 lên 4.40/5, và tăng trưởng thêm +0.30 cho điểm số `Completeness`. 

---

## 5. Nếu có thêm thời gian, em sẽ làm gì? (50-100 từ)

Em sẽ ứng dụng thêm tính năng **Xếp hạng lại (Reranking / Cross-Encoder)** vào cấu trúc Nâng cao thứ hai (Variant 2). Kết quả kiểm định LLM-as-a-Judge chỉ ra rằng dù Hybrid Search mang về đủ dòng thông tin, đôi khi nó lại gom dư thừa các đoạn khối dữ liệu gây nhiễu, làm tụt mốc tính gọn của câu trả lời. Nếu sử dụng Cross-encoder để đo duyệt hệ số đối chiếu (Query-Context Relevance Score) một vòng cuối cùng rồi loại trừ đoạn nhiễu trước khi nhồi sang khâu sinh (Generation), mức tính cực đoan hóa (Faithfulness) chắc chắn sẽ gắt gao và hoàn hảo hơn bao giờ hết.
