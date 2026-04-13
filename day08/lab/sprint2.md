# Sprint 2 Roadmap: Baseline RAG Implementation (Senior AI Engineer View)

Chào em, với tư cách là Senior AI Engineer, anh đã thiết kế lộ trình thực hiện Sprint 2 để em có thể triển khai song song trong khi đồng đội đang hoàn thiện Sprint 1 (Indexing). 

Mục tiêu của Sprint 2 là xây dựng một **Grounded RAG Pipeline**: Chỉ trả lời dựa trên bằng chứng, có trích dẫn nguồn, và biết nói "không" khi thiếu dữ liệu.

---

## 🛠 Prerequisites (Giả định từ Sprint 1)

Để Sprint 2 chạy được, chúng ta thống nhất các "interface" sau với Sprint 1:
- **Collection Name**: `rag_lab`
- **Embedding Model**: OpenAI `text-embedding-3-small`
- **DB Path**: Thư mục `chroma_db` tại root project.

---

## 1. Module 1: Dense Retrieval (`retrieve_dense`)

Phần này chịu trách nhiệm chuyển câu hỏi của người dùng thành vector và tìm kiếm các đoạn văn bản (chunks) có ý nghĩa tương đồng nhất trong ChromaDB.

### Code Snippet cho `rag_answer.py`:
```python
import chromadb
from index import get_embedding, CHROMA_DB_DIR

def retrieve_dense(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    # 1. Khởi tạo client và collection
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection("rag_lab")

    # 2. Embed câu hỏi người dùng
    query_embedding = get_embedding(query)

    # 3. Truy vấn vector store
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # 4. Format lại kết quả trả về
    formatted_results = []
    if results["documents"]:
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]  # Chuyển l2 distance sang similarity score
            })
    
    return formatted_results
```

---

## 2. Module 2: Generation (`call_llm`)

Sử dụng OpenAI Chat Completion để sinh câu trả lời. Anh khuyến nghị dùng `temperature=0` để đảm bảo tính ổn định (deterministic) khi đánh giá (evaluation).

### Code Snippet cho `rag_answer.py`:
```python
from openai import OpenAI

def call_llm(prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # Bắt buộc để output ổn định
        max_tokens=1024,
    )
    
    return response.choices[0].message.content
```

---

## 3. Module 3: Grounded Prompting Strategy

Đây là "linh hồn" của RAG. Prompt phải ép Model tuân thủ 3 quy tắc vàng:
1. **Evidence-only**: Không tự bịa (hallucination).
2. **Abstain**: Nói "Tôi không biết" nếu context không đủ.
3. **Citation**: Luôn kèm số thứ tự nguồn [1], [2].

Hàm `build_grounded_prompt` trong code hiện tại đã khá ổn, nhưng em có thể tinh chỉnh thêm phần **Instruction** để mạnh mẽ hơn:

> "Answer the user question ONLY based on the provided Context. If the information is not in the context, strictly respond: 'Dựa trên tài liệu hiện có, tôi không có đủ thông tin để trả lời câu hỏi này.' Do not use external knowledge."

---

## 4. Kiểm thử & Định nghĩa Hoàn thành (DoD)

Sau khi ghép nối các hàm vào `rag_answer()`, em cần kiểm tra với 3 kịch bản:

| Kịch bản | Câu hỏi ví dụ | Kết quả mong đợi |
|----------|---------------|------------------|
| **Happy Path** | "SLA P1 là bao lâu?" | Trả lời đúng, có kèm nguồn `[1]` hoặc `[source_name]` |
| **Abstain (Từ chối)** | "Bữa trưa hôm nay có gì?" | Trả lời: "Không đủ thông tin" |
| **Citation Check** | (Câu hỏi bất kỳ) | Metadata `sources` trong output không được rỗng |

---

## 💡 Senior Tips:
- **Latent Error**: Nếu `retrieve_dense` trả về toàn kết quả không liên quan, hãy kiểm tra lại xem `get_embedding` ở Sprint 1 và Sprint 2 có dùng chung 1 model không.
- **Cost Optimization**: Sprint 2 dùng `top_k_select = 3` là điểm cân bằng tốt nhất giữa độ chính xác và chi phí token.

Chúc em triển khai Sprint 2 thành công! Có vấn đề gì cứ gọi anh.
