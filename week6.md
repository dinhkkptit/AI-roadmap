Dưới đây là **Kiến thức chi tiết Tuần 6 – Supervised Fine-Tuning (SFT)** trong Roadmap học LLM & GenAI.  
Đây là một trong những phần quan trọng nhất nếu bạn muốn **tinh chỉnh LLM để phục vụ ứng dụng thực tế** (chatbot, RAG, hướng dẫn, task-specific model).

---

# 🟦 **TUẦN 6 — Supervised Fine-Tuning (SFT)**

Tuần này bạn học:

1. SFT là gì?
    
2. Dữ liệu SFT (instruction format)
    
3. Multi-turn conversation
    
4. Cách chuẩn hóa dữ liệu
    
5. Tokenization trong training
    
6. Hyperparameters quan trọng
    
7. Kiến trúc training LLM (Trainer / PEFT / DeepSpeed)
    
8. Evaluate model sau SFT
    
9. Thực hành pipeline SFT
    

---

# 🧠 **1. SFT là gì?**

**SFT (Supervised Fine-Tuning)** = huấn luyện LLM bằng dữ liệu có **đầu vào + đầu ra chuẩn** (supervised).

Giống như:

`instruction → input → expected answer`

SFT giúp mô hình:

- làm theo hướng dẫn (instruction-following)
    
- trả lời lịch sự, có cấu trúc
    
- phù hợp với task domain
    
- biết cách format output
    
- giảm hallucination
    

Trong OpenAI, Anthropic, Meta → **SFT là bước 1 trong RLHF pipeline**.

---

# 📄 **2. Dữ liệu SFT: Instruction Format**

Chuẩn phổ biến nhất hiện nay:

`[   {     "instruction": "Dịch câu sau sang tiếng Anh",     "input": "Xin chào",     "output": "Hello"   },   {     "instruction": "Giải thích mô hình LoRA",     "output": "LoRA là..."   } ]`

Các dạng format khác:

- ChatML (used in Qwen)
    
- Alpaca-style
    
- LLaMA 3 chat format
    
- ShareGPT multi-turn format
    

---

# 💬 **3. Multi-Turn Conversation (chat)**

LLM ngày nay dùng kiến trúc chat → cần dữ liệu nhiều lượt:

`[   {"role": "user", "content": "xin chào"},   {"role": "assistant", "content": "chào bạn"},   {"role": "user", "content": "hôm nay trời thế nào?"},   {"role": "assistant", "content": "trời đẹp"} ]`

SFT chat giúp LLM:

- nhớ bối cảnh
    
- giữ phong cách trò chuyện
    
- không quên lịch sử hội thoại
    

---

# 🧹 **4. Chuẩn hóa dữ liệu SFT**

Các bước thực tế:

### ✔ 4.1. Loại bỏ dữ liệu bẩn

- câu vô nghĩa
    
- câu dịch không chuẩn
    
- thông tin sai lệch
    

### ✔ 4.2. Cân bằng dữ liệu

- không để 1 tác vụ chiếm 90% dataset
    

### ✔ 4.3. Chuẩn hóa ngôn ngữ

- viết đúng chính tả
    
- format consistent
    

### ✔ 4.4. Giới hạn độ dài

- tách câu dài > 4096 tokens
    

---

# 🔤 **5. Tokenization trong training**

Phải decode dữ liệu theo đúng tokenizer của model:

- LLaMA tokenizer
    
- Qwen tokenizer
    
- Mistral tokenizer
    
- Tiktoken for GPT-like
    

### Cảnh báo:

❌ Không token hóa bằng tokenizer khác model  
❌ Multi-lingual → phải kiểm tra token splitting

---

# 🧮 **6. Hyperparameters quan trọng**

|Tham số|Gợi ý (SFT 7B–13B)|
|---|---|
|learning_rate|2e-5 → 5e-5|
|batch_size|1–8 (LoRA/QLoRA)|
|max_seq_len|1024–4096|
|epochs|1–3|
|warmup_steps|50–200|
|weight_decay|0.0|
|gradient_accumulation|4–32|
|LoRA rank|8–32|

✔ Với QLoRA:

`learning_rate = 2e-4 lora_alpha = 16–64`

---

# 🏗️ **7. Training pipeline**

### Bộ khung chuẩn:

1. Tokenize dataset
    
2. Format dataset theo ChatML / Alpaca
    
3. Load model (FP16 cho LoRA, NF4 cho QLoRA)
    
4. Chèn LoRA adapters
    
5. Setup Trainer (HuggingFace)
    
6. Training + evaluation
    
7. Merge LoRA (tùy chọn)
    
8. Export model
    

### Công cụ phổ biến:

- **LlamaFactory** (UI/CLI, dễ nhất)
    
- **Axolotl** (dùng phổ biến trong công ty)
    
- **PEFT + Transformers** (tự code, linh hoạt nhất)
    
- **DeepSpeed / Accelerate** (tối ưu GPU lớn)
    

---

# 📊 **8. Evaluate model sau SFT**

### Cách test:

- test với prompt từ dataset
    
- test ngoài dataset để generalize
    
- dùng NLP metrics: BLEU, ROUGE
    
- dùng đánh giá LLM: GPT-4 Judge
    
- test hallucination
    
- test long context
    

### RAG-specific SFT:

- đảm bảo model dùng context tốt
    
- không trả lời sai nếu không có thông tin
    

---

# 🧪 **9. Ví dụ thực tế: SFT Qwen2 7B bằng QLoRA**

### Đoạn code mẫu (Python + Transformers):

`from transformers import AutoModelForCausalLM, AutoTokenizer from peft import LoraConfig, get_peft_model from datasets import load_dataset  model_name = "Qwen/Qwen2-7B"  tokenizer = AutoTokenizer.from_pretrained(model_name) model = AutoModelForCausalLM.from_pretrained(     model_name,     load_in_4bit=True,     device_map="auto" )  lora = LoraConfig(     r=16,     lora_alpha=32,     target_modules=["q_proj","v_proj"],     lora_dropout=0.05 )  model = get_peft_model(model, lora) dataset = load_dataset("json", data_files="sft.json")  # training arguments ...`

Đây là một pipeline đầy đủ giúp bạn fine-tune thực sự.

---

# 🧠 **10. Best Practices (rất quan trọng)**

✔ Không dùng learning rate quá cao  
✔ Không train quá lâu → model overfit, trả lời kỳ quặc  
✔ Luôn test model trên prompt chưa thấy trong dataset  
✔ Format rõ ràng:

`<|user|>  <|assistant|>`

✔ Chỉ mở LoRA ở các layer attention → quality ổn định  
✔ Dữ liệu phải sạch và phù hợp task

---

# 🎁 **Tóm tắt Tuần 6**

|Mục|Hiểu gì?|
|---|---|
|SFT|Fine-tune theo dữ liệu giám sát|
|Dữ liệu SFT|Instruction, input, output|
|Conversation|Multi-turn chat|
|Tokenization|Rất quan trọng|
|Hyperparams|LR, batch, seq_len|
|Training tools|LlamaFactory, Axolotl, PEFT|
|Evaluate|Test chất lượng sau training|

---

# ❓ Bạn muốn tiếp gì tiếp theo?

- Xuất **Tuần 6** thành file `.md`
    
- Làm ví dụ dataset hoàn chỉnh
    
- Tạo code full training (Axolotl / PEFT / LlamaFactory)
    
- Tư vấn hyperparameters cho GPU của bạn
    
- Tạo lesson quiz bài kiểm tra tuần 6
    

Chỉ cần nói nhé!
