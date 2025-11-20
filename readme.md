# 🚀 **ROADMAP HỌC GEN AI & LLM (3–6 THÁNG)**

Tập trung cho mục tiêu: **Kỹ sư LLM / GenAI Engineer / RAG Engineer**

---

# 🟦 **THÁNG 1 — NỀN TẢNG LLM & GEN AI**

## 🎯 Mục tiêu tháng 1:

- Hiểu transformer và nguyên lý hoạt động của LLM
    
- Nắm cơ bản Machine Learning & Deep Learning
    
- Hiểu Attention, QKV, Tokenization, KV Cache
    

---

## **Tuần 1: Machine Learning nền tảng**

- Linear Algebra (vector, matrix, dot product)
    
- Probability (distribution, expectation)
    
- Optimization (SGD, Adam)
    
- Loss functions
    

👉 _Output:_ Bạn hiểu mô hình học như thế nào.

---

## **Tuần 2: Deep Learning**

- Neural Networks
    
- Backpropagation
    
- Overfitting & Regularization
    
- Activation: ReLU, GELU
    
- BatchNorm vs LayerNorm
    

👉 _Output:_ Nền tảng để hiểu Transformer.

---

## **Tuần 3: Transformer Core**

- Paper: “Attention is All You Need”
    
- Multi-Head Attention
    
- Q / K / V
    
- Positional Encoding & RoPE
    
- Feed-forward block
    
- Residual connections
    

👉 _Output:_ Bạn có thể giải thích Transformer cho người khác.

---

## **Tuần 4: LLM fundamentals**

- Why decoder-only architecture
    
- Autoregressive LM
    
- KV Cache (cực kỳ quan trọng)
    
- Tokenization (BPE, SentencePiece, Tiktoken)
    
- Comparing GPT vs Llama vs Mistral vs Qwen
    

👉 _Output:_ Bạn hiểu toàn bộ kiến trúc LLM.

---

# 🟩 **THÁNG 2 — THỰC HÀNH LLM: FINE-TUNING**

## 🎯 Mục tiêu tháng 2:

- Biết fine-tune LoRA, QLoRA
    
- Tự chạy model trên GPU
    
- Hiểu training: batch size, seq_len, VRAM
    

---

## **Tuần 5: LoRA & QLoRA**

- LoRA theory
    
- QLoRA (4bit quantization NF4)
    
- PEFT library
    
- Target modules (q_proj, v_proj…)
    
- Compute VRAM cho LoRA/QLoRA
    

👉 _Bài thực hành:_ Fine-tune Qwen2 1.5B bằng QLoRA.

---

## **Tuần 6: SFT (Supervised Fine-tuning)**

- Format dataset: instruction + input + output
    
- Tokenization trong training
    
- Dataset cleaning/dedup
    
- Trainer arguments
    
- Gradient Accumulation
    
- Checkpoint & evaluation
    

👉 _Bài thực hành:_ SFT model 7B với dataset tùy chọn.

---

## **Tuần 7: Training nâng cao**

- DeepSpeed ZeRO-2 / ZeRO-3
    
- FSDP (Fully Sharded Data Parallel)
    
- CUDA kernels (FlashAttention)
    
- Context length training
    

👉 _Bài thực hành:_ QLoRA Llama3 8B trên GPU 24GB.

---

## **Tuần 8: Preference Tuning**

- DPO
    
- ORPO
    
- PPO / RLHF (overview)
    

👉 _Bài thực hành:_ Tối ưu model bằng DPO.

---

# 🟧 **THÁNG 3 — RAG (RETRIEVAL-AUGMENTED GENERATION)**

## 🎯 Mục tiêu tháng 3:

- Thành thạo RAG
    
- Biết vector embedding, reranker
    
- Tự build 1 hệ thống RAG production-ready
    

---

## **Tuần 9: Embedding**

- Embedding models: bge-large, e5-large
    
- Vector representation
    
- Chunking strategies
    
- Token window size
    
- Metadata filtering
    

👉 _Bài thực hành:_ Index 10.000 docs vào FAISS/Qdrant.

---

## **Tuần 10: RAG cơ bản**

- RAG v1 architecture
    
- Retriever → LLM
    
- Prompt engineering
    
- Context window, KV cache ảnh hưởng RAG
    

👉 _Bài thực hành:_ Build RAG cho tài liệu công ty.

---

## **Tuần 11: Reranker (quant + cross-encoder)**

- bge-reranker-large
    
- Jina Reranker
    
- Multi-stage retrieval
    
- Pipeline: BM25 → embedding → reranker
    

👉 _Bài thực hành:_ RAG + reranker cho accuracy cao.

---

## **Tuần 12: Advanced RAG**

- Multi-query retrieval
    
- Query rewriting
    
- HyDE
    
- Graph RAG
    
- Agentic RAG
    

👉 _Bài thực hành:_ RAG tự tối ưu (intelligent retrieval).

---

# 🟨 **THÁNG 4 — LLM DEPLOYMENT & GPUs**

## 🎯 Mục tiêu tháng 4:

- Serve LLM bằng vLLM, TGI, TensorRT-LLM, Ollama
    
- Tối ưu VRAM
    
- Docker + Compose
    
- GPU sizing, MIG partition
    

---

## **Tuần 13: Inference Engine**

- So sánh: vLLM vs TGI vs TensorRT-LLM vs Ollama
    
- PagedAttention
    
- KV Cache management
    
- Continuous batching
    

👉 _Thực hành:_ Serve Qwen2 7B bằng vLLM + streaming.

---

## **Tuần 14: GPU Optimization**

- Batch size cho inference
    
- FlashAttention
    
- Quantization: INT4/8, GGUF
    
- Optimize throughput
    

👉 _Thực hành:_ Benchmark 7B, 13B, 70B trên GPU bạn có.

---

## **Tuần 15: Deployment**

- Dockerfile + Docker Compose
    
- GPU passthrough
    
- Multi-model hosting
    
- API design (OpenAI-compatible)
    

👉 _Thực hành:_ Triển khai Llama3 8B trên server riêng.

---

## **Tuần 16: Scaling & MLOps**

- K8s (Kubernetes)
    
- Horizontal autoscaling
    
- Model registry
    
- Monitoring (Prometheus + Grafana)
    
- Logging (ELK / Loki)
    

👉 _Thực hành:_ Deploy RAG + vLLM trên Kubernetes.

---

# 🟥 **THÁNG 5 — SECURITY, EVALUATION & PRODUCTION**

## 🎯 Mục tiêu tháng 5:

- Đưa mô hình vào sản xuất
    
- Đánh giá RAG + LLM
    
- Làm security / safety
    

---

## **Tuần 17: Security**

- Prompt injection
    
- Jailbreak
    
- Data leakage
    
- Safety filters
    
- Guardrails (Llama Guard, NeMo Guardrails)
    

👉 _Thực hành:_ Tạo guardrails cho chatbot.

---

## **Tuần 18: Evaluation**

- Perplexity
    
- MMLU
    
- MT-Bench
    
- RAGAS
    
- HELM
    

👉 _Thực hành:_ Benchmark model 7B và 13B.

---

## **Tuần 19: Monitoring production**

- Usage analytics
    
- Latency tracking
    
- Token cost estimation
    
- Versioning model API
    

👉 _Thực hành:_ Xây dashboard cho hệ thống.

---

## **Tuần 20: Optimization vòng đời**

- Distillation
    
- Knowledge editing
    
- Continual training
    
- Memory-augmented models
    

👉 _Thực hành:_ Distill 1 mô hình 7B thành 3B.

---

# 🟩 **THÁNG 6 — ADVANCED GEN AI (DÀNH CHO 6 THÁNG)**

## Tuần 21–24:

- Speculative decoding
    
- Medusa / EAGLE
    
- Multi-token prediction
    
- Agents (ReAct, Toolformer, CrewAI, AutoGen)
    
- Autonomous RAG
    
- MoE training
    
- FlashDecoding
    

👉 _Capstone Project:_  
**Build 1 hệ thống AI hoàn chỉnh:**

- LLM service
    
- RAG
    
- Reranker
    
- Monitoring
    
- Admin UI
    
- Fine-tuning pipeline
