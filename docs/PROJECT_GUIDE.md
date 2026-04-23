# 📘 Hướng dẫn Dự án: Legal Domain-Adaptive LLM Specialist in Vietnam

> Tài liệu hướng dẫn nội bộ — không cần đồng bộ lên GitHub.  
> Cập nhật lần cuối: 2026-03-02

---

## 1. Giới thiệu dự án

**Tên dự án:** Legal Domain-Adaptive LLM Specialist in Vietnam  
**Mục tiêu:** Xây dựng mô hình ngôn ngữ lớn (LLM) chuyên biệt cho lĩnh vực pháp luật Việt Nam, vượt trội hơn mô hình đa năng **25–40%** trên các bộ benchmark pháp lý.  
**Nền tảng chạy:** Kaggle Notebooks (miễn phí, $0)  
**Repository:** [github.com/vtgh04/Legal-LLM-Model](https://github.com/vtgh04/Legal-LLM-Model)

---

## 2. Môi trường thực thi (Kaggle)

Toàn bộ training, fine-tuning và inference **chỉ chạy trên Kaggle** do hạn chế phần cứng cục bộ.

### Phần cứng khả dụng

| GPU | Quota | Mục đích khuyến nghị |
|---|---|---|
| **T4 x2** | Không giới hạn | Fine-tuning QLoRA (mặc định) |
| **T4 x4** | **30h / tuần** | Training lớn — dùng tiết kiệm! |
| **P100** | Không giới hạn | Thay thế T4 x2 |
| **CPU** | Không giới hạn | EDA, kiểm tra dữ liệu nhẹ |

### Đường dẫn cố định (KHÔNG sửa)

```python
# Input (chỉ đọc)
INPUT_BASE = "/kaggle/input/datasets/giahuyvotran"

# Output (ghi được)
OUTPUT_BASE    = "/kaggle/working"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
MODEL_DIR      = "/kaggle/working/model"
LOG_DIR        = "/kaggle/working/logs"
```

> ⚠️ Mọi file đầu ra (checkpoint, model, log) phải lưu vào `/kaggle/working/`.  
> `/kaggle/input/` là **read-only**, tuyệt đối không ghi vào đây.

---

## 3. Quy tắc làm việc với Agent

| ✅ Agent SẼ làm | ❌ Agent KHÔNG làm |
|---|---|
| Thiết kế kiến trúc model | Chạy / execute cell |
| Viết script tiền xử lý dữ liệu | Cài đặt package tự động |
| Viết code training PyTorch / HuggingFace | Giả sử file tồn tại mà không kiểm tra |
| Xử lý lỗi và debug traceback | Dùng đường dẫn cứng ngoài Kaggle |
| Tối ưu code cho T4/P100 (4-bit quant) | |

**Bạn (người dùng)** là người chạy cell trên Kaggle. Agent cung cấp code hoàn chỉnh, sẵn sàng chạy.

---

## 4. Pipeline 8 giai đoạn

```
Stage 1 (CPT)   → Continued Pre-Training trên corpus luật tiếng Việt
Stage 2 (SFT)   → Supervised Fine-Tuning với legal instruction dataset
Stage 3 (PEFT)  → QLoRA / LoRA / DoRA (ưu tiên QLoRA cho T4)
Stage 4 (RLHF)  → DPO / ORPO / SimPO
Stage 5 (RAG)   → Dense + Sparse Hybrid Retrieval + citation graph
Stage 6 (Merge) → SLERP / TIES / DARE
Stage 7 (Quant) → 4-bit / 2-bit + Knowledge Distillation
Stage 8 (MoE)   → Mixture-of-Experts routing theo lĩnh vực luật
```

**Quyết định mặc định:** Bắt đầu từ **QLoRA (Stage 3)**. Chỉ chạy CPT (Stage 1) khi domain gap > 30% trên LegalBench zero-shot.

---

## 5. Cấu trúc thư mục

```
Legal-LLM-Model/
│
├── .agents/                    ← Tài liệu hướng dẫn cho agent AI
│   ├── KAGGLE_ENV.md           ← ⚠️ Đọc TRƯỚC (môi trường, I/O, quy tắc code)
│   ├── skills/
│   │   └── legal-llm-specialist/
│   │       └── SKILL.md        ← Đọc SAU (nhiệm vụ, năng lực, SOP)
│   └── workflows/
│       ├── data-pipeline.md    ← Quy trình xử lý dữ liệu (/data-pipeline)
│       ├── fine-tuning.md      ← Quy trình fine-tuning (/fine-tuning)
│       └── evaluation.md       ← Quy trình đánh giá (/evaluation)
│
├── configs/                    ← File YAML cấu hình training
│   └── sft_template.yaml
│
├── data/
│   ├── raw/                    ← Corpus luật gốc — KHÔNG chỉnh sửa
│   ├── processed/              ← Dataset đã làm sạch & tokenize
│   └── external/               ← Dữ liệu ngoài, benchmark
│
├── docs/                       ← Tài liệu kỹ thuật (bạn đang đọc file này)
│   └── PROJECT_GUIDE.md        ← ← ← BẠN ĐANG Ở ĐÂY
│
├── models/
│   ├── pretrained/             ← Base model tải về (VD: Qwen, LLaMA, Vistral)
│   └── checkpoints/            ← Adapter & weights sau fine-tuning
│
├── notebooks/                  ← Jupyter notebooks chạy trên Kaggle
│
├── scripts/                    ← Script tự động hóa (download, merge, push HF)
│
├── src/
│   ├── data_processing/        ← Làm sạch, chuẩn hóa, tokenize
│   ├── training/               ← SFT, DPO, CPT trainers
│   ├── evaluation/             ← Chạy benchmark
│   ├── inference/              ← Serving, prediction
│   └── utils/                  ← Hàm dùng chung, logging
│
└── tests/                      ← Unit & integration tests
```

---

## 6. Stack công nghệ

### Fine-Tuning
| Thư viện | Vai trò |
|---|---|
| `transformers` | Load model, tokenizer |
| `peft` | LoRA / QLoRA / DoRA |
| `trl` | SFT Trainer, DPO Trainer |
| `accelerate` | Multi-GPU training |
| `bitsandbytes` | 4-bit quantization |
| `unsloth` | Fine-tuning nhanh hơn (tùy chọn) |
| `datasets` | Load và xử lý dataset |

### RAG Pipeline
| Thư viện | Vai trò |
|---|---|
| `langchain` / `llama-index` | Orchestration RAG |
| `faiss` | Vector search (dense) |
| `qdrant` / `weaviate` | Vector store (nâng cao) |

### Theo dõi thực nghiệm
| Thư viện | Vai trò |
|---|---|
| `wandb` | Theo dõi loss, metrics |
| `mlflow` | Quản lý experiment |

---

## 7. Tiêu chuẩn KPI (Production-Ready)

| Metric | Ngưỡng tối thiểu | Alert khi |
|---|---|---|
| LegalBench Accuracy | **≥ 82%** | < 75% |
| CUAD Contract F1 | **≥ 92%** | < 85% |
| Citation Accuracy | **≥ 95%** | < 90% |
| Hallucination Rate | **≤ 2%** | > 5% |
| Inference Cost | **< $0.001 / 1k token** | > $0.003 |
| Thời gian PEFT → Production | **< 7 ngày** | > 14 ngày |

---

## 8. Quy trình làm việc điển hình

```
Bước 1: Chuẩn bị dữ liệu
  → Chạy workflow /data-pipeline
  → Output: /kaggle/working/data_processed/

Bước 2: Fine-tuning
  → Chọn base model (Qwen2.5, LLaMA3, Vistral...)
  → Chạy workflow /fine-tuning với QLoRA
  → Checkpoint lưu tại /kaggle/working/checkpoints/

Bước 3: Đánh giá
  → Chạy workflow /evaluation
  → So sánh với KPI dashboard

Bước 4: Nếu đạt KPI → Export model
  → Merge adapter vào base model
  → Push lên HuggingFace Hub hoặc lưu GGUF

Bước 5: RAG + Serving
  → Kết nối vector store
  → Deploy API hoặc Gradio demo
```

---

## 9. Quy ước về Memory trên T4/P100

Luôn áp dụng khi train model ≥ 7B:

```python
# Bắt buộc khi train trên Kaggle T4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Training arguments tối ưu cho T4
per_device_train_batch_size = 1   # Bắt đầu ở 1
gradient_accumulation_steps = 8   # Effective batch = 8
gradient_checkpointing = True     # Tiết kiệm VRAM
fp16 = True                       # Mixed precision
```

---

## 10. Guardrails bắt buộc

1. **Disclaimer**: Mọi câu trả lời pháp lý phải kèm *"Đây không phải tư vấn pháp lý. Hãy tham khảo luật sư có chuyên môn."*
2. **Jurisdiction**: Luôn xác định phạm vi pháp lý (Việt Nam / cụ thể tỉnh/thành) trước khi trả lời.
3. **Citation**: Mọi khẳng định pháp lý phải có nguồn (điều / khoản / văn bản luật).
4. **PII**: Không lưu trữ thông tin cá nhân trong log hoặc dataset.

---

*Cập nhật tài liệu này mỗi khi có thay đổi lớn về pipeline hoặc môi trường.*
