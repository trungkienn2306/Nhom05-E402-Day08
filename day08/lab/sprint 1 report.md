# Sprint 1 Report — Build RAG Index

**Nhóm:** 05 — E402  
**Ngày:** 13/04/2026  
**File chính:** `day08/lab/index.py`

---

## Mục tiêu Sprint 1

Xây dựng pipeline indexing hoàn chỉnh cho hệ thống RAG (Retrieval-Augmented Generation):
- Đọc và tiền xử lý tài liệu từ `data/docs/`
- Chia tài liệu thành các chunk nhỏ theo cấu trúc tự nhiên
- Gắn metadata đầy đủ cho mỗi chunk
- Embed và lưu vào vector store (ChromaDB)

---

## Tài liệu được index

| File | Mô tả |
|------|-------|
| `access_control_sop.txt` | Quy trình kiểm soát truy cập hệ thống |
| `hr_leave_policy.txt` | Chính sách nghỉ phép nhân sự |
| `it_helpdesk_faq.txt` | FAQ hỗ trợ IT |
| `policy_refund_v4.txt` | Chính sách hoàn tiền v4 |
| `sla_p1_2026.txt` | SLA ưu tiên P1 năm 2026 |

---

## Cấu hình

```python
CHUNK_SIZE    = 400   # tokens (~1600 ký tự)
CHUNK_OVERLAP = 80    # tokens (~320 ký tự)
```

---

## Chi tiết các bước đã thực hiện

### Bước 1 — Cài đặt môi trường

- Activate virtual environment `.venv`
- Cài đặt thư viện:
  ```bash
  pip install -r requirements.txt
  pip install openai chromadb
  ```
- Cấu hình `OPENAI_API_KEY` trong file `.env`

---

### Bước 2 — Implement `get_embedding()`

**Vấn đề ban đầu:** Hàm chỉ có `raise NotImplementedError(...)`, chưa gọi API.

**Giải pháp:** Dùng OpenAI Embeddings API với model `text-embedding-3-small`.

```python
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding
```

**Lý do chọn `text-embedding-3-small`:**
- Chi phí thấp, tốc độ nhanh
- Hỗ trợ tốt tiếng Việt và tiếng Anh
- Dimension 1536, đủ chất lượng cho RAG

---

### Bước 3 — Cải tiến `_split_by_size()`

**Vấn đề ban đầu:** Cắt chunk theo số ký tự cứng, có thể cắt giữa câu/từ.

**Giải pháp:** Thêm logic tìm ranh giới tự nhiên trước khi cắt.

```python
def _split_by_size(text, base_metadata, section,
                   chunk_chars=CHUNK_SIZE * 4,
                   overlap_chars=CHUNK_OVERLAP * 4):

    if len(text) <= chunk_chars:
        return [{"text": text, "metadata": {**base_metadata, "section": section}}]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))

        # Tìm ranh giới tự nhiên: '.', '\n', hoặc ' '
        if end < len(text):
            while end > start and text[end] not in ['.', '\n', ' ']:
                end -= 1
            if end == start:
                end = min(start + chunk_chars, len(text))

        chunks.append({
            "text": text[start:end].strip(),
            "metadata": {**base_metadata, "section": section},
        })
        start = end - overlap_chars  # Overlap giữa các chunk

    return chunks
```

**Cải tiến so với ban đầu:**
| | Trước | Sau |
|--|-------|-----|
| Điểm cắt | Cứng theo số ký tự | Tại `.`, `\n`, hoặc dấu cách gần nhất |
| Chunk text | Có thể có khoảng trắng thừa | Luôn `.strip()` |
| Overlap | Có | Có (giữ nguyên) |

---

### Bước 4 — Hoàn thiện `build_index()`

**Vấn đề ban đầu:** Toàn bộ logic ChromaDB bị comment, không lưu được vào DB.

**Giải pháp:** Implement đầy đủ pipeline 3 bước.

```python
def build_index(docs_dir=DOCS_DIR, db_dir=CHROMA_DB_DIR):
    import chromadb

    # 1. Khởi tạo ChromaDB — reset collection để tránh trùng lặp
    chroma_client = chromadb.PersistentClient(path=str(db_dir))
    try:
        chroma_client.delete_collection("rag_lab")
    except Exception:
        pass
    collection = chroma_client.create_collection(
        name="rag_lab",
        metadata={"hnsw:space": "cosine"}  # Dùng cosine similarity
    )

    # 2. Với mỗi file .txt trong docs_dir
    for filepath in docs_dir.glob("*.txt"):
        raw_text = filepath.read_text(encoding="utf-8")

        # Preprocess → Chunk → Embed → Store
        doc    = preprocess_document(raw_text, str(filepath))
        chunks = chunk_document(doc)

        for i, chunk in enumerate(chunks):
            chunk_id  = f"{filepath.stem}_{i}"
            embedding = get_embedding(chunk["text"])
            collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk["text"]],
                metadatas=[chunk["metadata"]],
            )
```

**Lưu ý kỹ thuật:**
- Dùng `upsert` thay vì `insert` để idempotent (chạy lại nhiều lần không bị lỗi trùng ID)
- `hnsw:space: cosine` — phù hợp với embedding text (không dùng L2)
- Đổi tên biến thành `chroma_client` để tránh conflict với `client` của OpenAI

---

### Bước 5 — Uncomment phần `__main__`

Bỏ comment 3 dòng trong `if __name__ == "__main__":`:

```python
build_index()             # Chạy pipeline đầy đủ
list_chunks()             # In 5 chunk đầu để kiểm tra
inspect_metadata_coverage()  # Thống kê metadata
```

---

## Chiến lược Chunking

Pipeline chia làm **2 tầng**:

```
Tài liệu (.txt)
    │
    ▼
Tầng 1: Split theo === Section === heading
    │     → Giữ ranh giới ngữ nghĩa tự nhiên
    │
    ▼
Tầng 2: Nếu section > 1600 ký tự → split theo ký tự
          + Tìm ranh giới gần nhất (. / \n / space)
          + Overlap 320 ký tự (~80 tokens) giữa các chunk
```

---

## Metadata được lưu cho mỗi chunk

| Field | Nguồn | Ví dụ |
|-------|-------|-------|
| `source` | Header file | `it/access-control-sop.md` |
| `section` | Heading `=== ... ===` | `Section 2: Phân cấp quyền truy cập` |
| `department` | Header file | `IT Security` |
| `effective_date` | Header file | `2026-01-01` |
| `access` | Header file | `internal` |

---

## Vấn đề gặp phải & cách giải quyết

| Vấn đề | Nguyên nhân | Giải pháp |
|--------|------------|-----------|
| `ModuleNotFoundError: No module named 'openai'` | Venv chưa cài openai | `pip install openai` |
| `ModuleNotFoundError: No module named 'chromadb'` | Venv chưa cài chromadb | `pip install chromadb` |
| `Python was not found` | Chạy sai thư mục, chưa activate venv | `cd day08/lab` → activate `.venv` |
| `chromadb.errors.NotFoundError` khi delete collection | Collection chưa tồn tại lần đầu chạy, guide bắt `ValueError` nhưng chromadb mới raise `NotFoundError` | Đổi `except ValueError` → `except Exception` |

---

## Definition of Done — Checklist

- [x] Script `index.py` chạy được không lỗi
- [x] Index đủ 5 tài liệu `.txt`
- [x] Mỗi chunk có ít nhất 3 metadata fields hữu ích: `source`, `section`, `effective_date`
- [x] `list_chunks()` in ra đúng thông tin
- [x] `inspect_metadata_coverage()` thống kê được theo `department`
- [x] Chunking không cắt giữa câu (dùng ranh giới tự nhiên)
- [x] Vector store dùng cosine similarity
