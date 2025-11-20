# 🟦 **TUẦN 8: RAG (Retrieval-Augmented Generation)**

Tuần này cực kỳ quan trọng vì **80% hệ thống AI thực tế** (chatbot doanh nghiệp, trợ lý luật, trợ lý y tế, hệ thống tìm kiếm thông minh…) đều dựa vào **RAG**.  
Bạn sẽ học toàn bộ pipeline: từ **Embedding, Chunking, Retriever, Reranker, Context Builder, Generator**, đến **Evaluation & Optimization**.

---

# 🚀 **TỔNG QUAN RAG (Retrieval-Augmented Generation)**

**RAG = mô hình LLM + công cụ truy xuất dữ liệu (retriever)**  
Giúp LLM trả lời dựa trên dữ liệu thật (PDF, web, tài liệu nội bộ) thay vì đoán.

> 💡 RAG = Search + LLM  
> → Tiết kiệm chi phí  
> → Không cần fine-tune nhiều  
> → Giảm hallucination

---

# 🟦 **1. Kiến trúc tổng thể của RAG**

Pipeline chuẩn:

`User Query      ↓ Query Preprocessing      ↓ Embedding Model      ↓ Vector Database (retriever)      ↓ Reranker (re-rank top K)      ↓ Context Builder (chunk assembly)      ↓ LLM Generator      ↓ Final Answer`

RAG hiện đại = **Retriever + Reranker + Generator**.

---

# 🟩 **2. Embedding (Vector Representation)**

Embedding là trái tim của RAG.

## 2.1. Loại embedding cho RAG:

✔ Text embedding  
✔ Multi-lingual embedding  
✔ Document embedding  
✔ Query embedding

## 2.2. Mô hình embedding tốt nhất hiện nay:

- **BGE-M3 (SOTA)**
    
- BGE large
    
- E5-Mistral
    
- Instructor-XL
    
- GTE Large
    
- mContriever (Meta)
    

## 2.3. Yêu cầu embedding tốt:

- Semantic similarity cao
    
- Đa ngôn ngữ
    
- Zero-shot robust
    
- Không drift qua domain
    

---

# 🟦 **3. Chunking (Tách tài liệu)**

Nếu chunk sai → RAG thất bại.

## 3.1. Các chiến lược chunking:

- **Fixed-size** (512–1024 chars)
    
- **Recursive Chunking** (tốt nhất cho PDF)
    
- **Semantic Chunking** (chia theo topic)
    
- **Windowed Chunking** (overlap: 50–100 tokens)
    

## 3.2. Best practice:

- Chunk size: **512–1024 tokens**
    
- Overlap: **50–150 tokens**
    
- Dùng LangChain “RecursiveCharacterTextSplitter”
    

---

# 🟧 **4. Vector Database (Retriever)**

Nơi lưu trữ embedding của tài liệu.

## 4.1. Vector DB phổ biến:

- **FAISS** (nhanh nhất, offline)
    
- **Milvus**
    
- **Weaviate**
    
- **Qdrant**
    
- **Pinecone** (SaaS)
    

## 4.2. Index phổ biến:

- HNSW
    
- IVF
    
- FlatIP
    
- ScaNN
    

## 4.3. Các kỹ thuật tăng chất lượng:

- Re-ranking
    
- Hybrid search (keyword + vector)
    
- kNN threshold
    
- Filter metadata
    

---

# 🟦 **5. Retriever (Lấy dữ liệu)**

Đầu vào = query → embedding  
Đầu ra = top K chunks

Các phương pháp retriever:

- Dense retriever (embedding-based)
    
- Sparse retriever (BM25)
    
- Hybrid (BM25 + Embedding) ← mạnh nhất
    

---

# 🟥 **6. Reranker (Cực quan trọng – tăng độ chính xác)**

Reranker = mô hình cross-encoder đánh giá lại từng chunk xem có liên quan thật không.

## Reranker tốt nhất:

- **BGE-Reranker-Large**
    
- **Cohere Reranker**
    
- **ColBERTv2**
    
- **Qwen2-Reranker**
    

## Cách hoạt động:

Trong retriever:

`Embedding(query) → retrieve top 20`

Trong reranker:

`CrossEncoder(query, chunk) → score Top 5 (after re-ranking)`

➡ Tăng độ chính xác 30–60%.

---

# 🟦 **7. Context Builder (Xây dựng bối cảnh)**

LLM rất nhạy cảm với **context format**.

## Chiến lược:

- Sorting theo điểm liên quan
    
- Chunk merging
    
- Context window tối ưu (4k–128k)
    
- Prompt template chuẩn RAG
    
- Citations
    

Ví dụ prompt RAG tốt:

`You are a retrieval-based assistant. Use ONLY the provided context to answer. If not found, say "I don't know".  Context: {{documents}}  Question: {{query}}`

---

# 🟩 **8. Generator (LLM tạo câu trả lời)**

Bạn có thể dùng:

- LLaMA 3
    
- Qwen 2
    
- Mistral
    
- Gemma
    
- GPT-4/4o nếu cần chất lượng cao
    

---

# 🟦 **9. RAG nâng cao (Advanced RAG)**

## 9.1. RAG Fusion

- Tạo nhiều query từ 1 query
    
- Tăng khả năng tìm đúng
    

## 9.2. HyDE (Hypothetical Document Embedding)

LLM tạo tài liệu “giả lập” về query → embedding → retrieve  
➡ cải thiện accuracy cực mạnh.

## 9.3. Query Rewriting

LLM cải thiện câu hỏi người dùng.

## 9.4. Multi-vector Retrieval

Dùng nhiều embedding cho một document (passage-level).

## 9.5. Graph RAG

Xây knowledge graph → traverse theo quan hệ.

## 9.6. Long-context RAG

Dùng LLM context 128k – 1M → bỏ vector DB (chỉ với tài liệu <1M tokens).

---

# 🟥 **10. Đánh giá RAG (RAG Evaluation)**

## 10.1. Metrics:

- **RAGAS** (SOTA)
    
- Faithfulness
    
- Answer relevance
    
- Context recall
    
- Context precision
    
- Citation accuracy
    

## 10.2. LLM-based evaluation

Dùng GPT-4o, Claude 3 để đánh giá output.

---

# 🟦 **11. Công cụ RAG hiện đại**

|Tool|Mạnh về|
|---|---|
|**LlamaIndex**|dựng RAG end-to-end|
|**LangChain**|pipeline linh hoạt|
|**Haystack**|retriever chuyên nghiệp|
|**Chroma**|local vector DB|
|**Milvus/Qdrant**|production vector DB|

---

# 🎯 **Tóm tắt Tuần 8**

|Thành phần|Vai trò|
|---|---|
|Embedding|biểu diễn câu thành vector|
|Chunking|chia tài liệu hợp lý|
|Retriever|tìm top K documents|
|Reranker|xếp lại chất lượng|
|Generator|LLM tạo câu trả lời|
|Context builder|tổ chức dữ liệu|
|Advanced RAG|Fusion, HyDE, Query Rewrite|
|Evaluation|RAGAS, LLM judge|
