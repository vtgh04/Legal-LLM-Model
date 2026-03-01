# Kaggle Environment — Agent Operating Notes

> **⚠️ MANDATORY READ**: Every agent working on this project MUST read this file
> before writing any code, notebook cell, or script. These rules are non-negotiable.

---

## 1. Why Kaggle?

Due to **local hardware limitations**, ALL training, fine-tuning, and inference
runs for this project are executed **exclusively on Kaggle Notebooks** at **zero cost ($0)**.

Do NOT write code that assumes a local GPU, local Docker container, or any
cloud provider (AWS, GCP, Azure) unless explicitly told otherwise.

---

## 2. Hardware Budget

| Resource | Availability | Notes |
|---|---|---|
| **GPU T4 x2** | ✅ Free | Primary training accelerator |
| **GPU P100** | ✅ Free | Alternative; single GPU |
| **GPU T4 x4** | ⚠️ Limited | 30 hours / week quota — use only for large runs |
| **No GPU** | ✅ Free | CPU-only; use for data inspection and lightweight preprocessing |
| **RAM** | ~13–30 GB | Varies by session type |
| **Disk** | ~20 GB (`/kaggle/working/`) | Outputs only; inputs are read-only |

**Strategy:**
- Use **T4 x2** by default for fine-tuning (QLoRA / LoRA).
- Reserve **T4 x4** quota for final full training runs only.
- Use CPU sessions for dataset inspection, EDA notebooks, and quick debug runs.
- Always enable **4-bit quantization (QLoRA / bitsandbytes)** to fit models into T4 VRAM (16 GB per card).

---

## 3. Strict I/O Path Rules

These paths are **hardcoded** and must appear verbatim in all code.

### 📥 Input (READ-ONLY — NEVER write here)

```python
INPUT_BASE = "/kaggle/input/datasets/giahuyvotran"
```

All datasets must be referenced under this base path. Example:

```python
TRAIN_DATA   = "/kaggle/input/datasets/giahuyvotran/train.jsonl"
CORPUS_DIR   = "/kaggle/input/datasets/giahuyvotran/corpus/"
```

> **Note:** The input path is read-only on Kaggle. Any attempt to write to
> `/kaggle/input/` will fail. All writes must go to `/kaggle/working/`.

### 📤 Output (WRITABLE)

```python
OUTPUT_BASE  = "/kaggle/working"
MODEL_DIR    = "/kaggle/working/model"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
LOG_DIR      = "/kaggle/working/logs"
```

All model checkpoints, processed datasets, evaluation results, and logs must be
saved under `/kaggle/working/`. Kaggle automatically zips and stores everything
in this directory at the end of a session.

---

## 4. Agent Coding Rules (STRICTLY ENFORCED)

### ✅ Agent WILL do:
- **Design** the model architecture and training pipeline.
- **Write** full, runnable Python scripts and Jupyter notebook cells.
- **Write** data preprocessing scripts for Vietnamese legal corpora.
- **Write** training code using PyTorch and HuggingFace (Transformers, PEFT, TRL, Accelerate).
- **Write** evaluation and inference scripts.
- **Debug** and fix code errors (tracebacks, shape mismatches, OOM errors, etc.).
- **Optimize** code for T4/P100 GPU constraints (gradient checkpointing, mixed precision, 4-bit quant).

### ❌ Agent will NOT do:
- **RUN** or execute any cell / script.
- **Install** packages (write `!pip install ...` cells for the user to run).
- Assume any file exists before checking with `os.path.exists()`.
- Hard-code absolute paths other than the Kaggle paths defined above.
- Use more than **4 GB VRAM** per model shard without explicit 4-bit quantization.

---

## 5. Standard Notebook Cell Template

Every notebook the agent produces must follow this header pattern:

```python
# ─── Environment Check ────────────────────────────────────────────────────────
import os, sys
print("Python:", sys.version)
print("Working dir:", os.getcwd())
print("GPU available:", os.popen("nvidia-smi --query-gpu=name --format=csv,noheader").read().strip())

# ─── Paths (DO NOT CHANGE) ────────────────────────────────────────────────────
INPUT_BASE     = "/kaggle/input/datasets/giahuyvotran"
OUTPUT_BASE    = "/kaggle/working"
CHECKPOINT_DIR = f"{OUTPUT_BASE}/checkpoints"
LOG_DIR        = f"{OUTPUT_BASE}/logs"
MODEL_DIR      = f"{OUTPUT_BASE}/model"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
print("Directories ready ✓")
```

---

## 6. Recommended Package Stack (Kaggle-compatible)

Write `!pip install` cells at the top of every notebook. Do NOT assume packages
are pre-installed beyond the Kaggle base image.

```python
# Cell 1 — Install dependencies (user must run this cell first)
!pip install -q \
    transformers>=4.40.0 \
    peft>=0.10.0 \
    trl>=0.8.0 \
    accelerate>=0.28.0 \
    bitsandbytes>=0.43.0 \
    datasets>=2.18.0 \
    evaluate>=0.4.1 \
    sentencepiece>=0.2.0 \
    wandb
```

---

## 7. Memory Management Rules for T4 / P100

Always apply these settings for any training run:

```python
from transformers import BitsAndBytesConfig
import torch

# 4-bit quantization config (mandatory for 7B+ models on T4)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Training args — always enable these on Kaggle
training_args = dict(
    fp16=True,                        # Mixed precision
    gradient_checkpointing=True,      # Save VRAM at cost of speed
    per_device_train_batch_size=1,    # Start low; increase if no OOM
    gradient_accumulation_steps=8,    # Effective batch = 8
    dataloader_num_workers=2,
    save_strategy="steps",
    save_steps=100,
    output_dir=CHECKPOINT_DIR,
)
```

---

## 8. Checkpoint & Model Saving Policy

- Save a checkpoint **every 100 steps**.
- Always run `model.save_pretrained(MODEL_DIR)` and
  `tokenizer.save_pretrained(MODEL_DIR)` at the **end of training**.
- Push to HuggingFace Hub using `push_to_hub=True` if a HF token is available.

```python
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"Model saved to {MODEL_DIR} ✓")
```

---

## 9. Debugging Protocol

When an error occurs in a Kaggle notebook, the agent must:

1. Read the **full traceback** carefully.
2. Check for **OOM (Out of Memory)**: reduce `per_device_train_batch_size` to 1 and enable `gradient_checkpointing`.
3. Check for **shape mismatches**: print `input_ids.shape`, `attention_mask.shape`, `labels.shape`.
4. Check for **path errors**: verify `/kaggle/input/datasets/giahuyvotran/...` exists with `os.path.exists()`.
5. Provide a **corrected code cell** only — do not re-run.
