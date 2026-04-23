# Roadmap: Domain-Adaptive Legal LLM for Vietnam

**Version:** 1.0
**Last updated:** 2026-03-02
**Scope:** Vietnamese legal domain — Bộ luật Dân sự, Bộ luật Hình sự, Nghị định, Thông tư, Bản án Tòa án Nhân dân
**Target:** MVP within 4–9 weeks using exclusively free resources (Kaggle Notebooks, Google Colab, HuggingFace free tier)
**Total cost:** $0

---

## Overview

This roadmap defines a structured, end-to-end process for building a domain-adaptive large language model specialized in Vietnamese law. The pipeline follows eight sequential technical stages: data acquisition and preprocessing, continued pre-training, supervised fine-tuning, preference optimization, retrieval-augmented generation, model merging, quantization, and evaluation. Each stage has explicit inputs, outputs, tooling decisions, and acceptance criteria.

This document is the primary reference for agents executing tasks on this project. Every code action must trace back to a stage defined here.

---

## Stage 0: Environment Setup

**Duration:** Day 1
**Platform:** Kaggle Notebooks

### Objective

Standardize the compute environment so that every subsequent notebook runs identically.

### Fixed I/O Paths

```python
INPUT_BASE     = "/kaggle/input/datasets/giahuyvotran"
OUTPUT_BASE    = "/kaggle/working"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
MODEL_DIR      = "/kaggle/working/model"
LOG_DIR        = "/kaggle/working/logs"
PROCESSED_DIR  = "/kaggle/working/data_processed"
```

### Standard Notebook Header

Every notebook must begin with this block, unmodified:

```python
import os, sys, torch
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

for d in [CHECKPOINT_DIR, MODEL_DIR, LOG_DIR, PROCESSED_DIR]:
    os.makedirs(d, exist_ok=True)
```

### Dependency Installation Cell

```python
!pip install -q \
    transformers>=4.40.0 \
    peft>=0.10.0 \
    trl>=0.8.0 \
    accelerate>=0.28.0 \
    bitsandbytes>=0.43.0 \
    datasets>=2.18.0 \
    evaluate>=0.4.1 \
    sentencepiece>=0.2.0 \
    faiss-cpu>=1.8.0 \
    unsloth \
    wandb \
    underthesea
```

### Acceptance Criteria

- GPU is detected and named (T4, P100, or T4 x2)
- All output directories created without error
- All packages install without conflict

---

## Stage 1: Data Acquisition

**Duration:** Week 1
**Notebook:** `notebooks/01_data_acquisition.ipynb`

### Objective

Collect all Vietnamese legal text corpora required for continued pre-training, instruction fine-tuning, preference optimization, and RAG indexing.

### Primary Sources (Kaggle Datasets — one-click download)

| Dataset                           | Kaggle Path                                                                          | Size    | Use               |
| --------------------------------- | ------------------------------------------------------------------------------------ | ------- | ----------------- |
| Zalo AI 2021 Legal Text Retrieval | `hariwh0/zaloai2021-legal-text-retrieval` or `lookingformyself/zac2021-ltr-data` | ~160 MB | CPT corpus + RAG  |
| Vietnamese Legal Corpus           | `quangbut/vietnamese-legal`                                                        | ~896 MB | CPT supplementary |
| Vietnamese Legal Dataset          | `thinh4526/vietnamese-legal`                                                       | ~200 MB | CPT supplementary |
| Vietnamese Legal Dataset          | `tuongdang/vietnamese-legal-dataset`                                               | varies  | SFT seed data     |

### Secondary Sources (HuggingFace — load via `datasets` library)

```python
from datasets import load_dataset

# Legal Q&A pairs for SFT
qa_dataset = load_dataset("thangvip/vietnamese-legal-qa")

# ShareGPT-format chat for SFT
chat_dataset = load_dataset("luanngo/Vietnamese-Legal-Chat-Dataset")

# Synthetic queries for RAG training (500,000+ queries)
query_dataset = load_dataset("phamson02/large-vi-legal-queries")

# Expert-annotated evaluation set (gold standard, 3,129 questions)
eval_dataset = load_dataset("ntphuc149/ViBidLQA_v1")
```

### Tertiary Sources (Web — RAG corpus only, not for training weights)

- `vbpl.vn` and `thuvienphapluat.vn` — published laws and decrees (scrape for RAG index, do not use for training to avoid licensing issues)
- `congbobanan.toaan.gov.vn` — published Supreme People's Court judgments

### Dataset Size Targets

| Split                | Minimum token count    | Format                                         |
| -------------------- | ---------------------- | ---------------------------------------------- |
| CPT corpus           | 200M tokens            | Plaintext, one document per line               |
| SFT training         | 10,000–50,000 samples | JSONL:`{instruction, input, output}`         |
| DPO preference pairs | 2,000–10,000 pairs    | JSONL:`{prompt, chosen, rejected}`           |
| RAG index            | 224,000+ passages      | JSONL:`{id, text, metadata}`                 |
| Evaluation           | 3,000+ annotated QA    | JSONL:`{question, reference_answer, source}` |

### Acceptance Criteria

- `legal_corpus.json` from Zalo AI extracted and readable
- HuggingFace datasets load without error
- Token count of combined CPT corpus logged to `LOG_DIR/data_stats.json`

---

## Stage 2: Data Preprocessing

**Duration:** Week 1
**Notebook:** `notebooks/02_data_preprocessing.ipynb`
**Refactored code:** `src/data_processing/`

### Objective

Transform raw legal text into clean, structured, training-ready datasets. Vietnamese legal text requires domain-specific handling that general-purpose cleaners do not provide.

### Step 2.1: Unicode Normalization

Vietnamese text often contains mixed Unicode representations of diacritics. Normalize to NFC before any further processing.

```python
import unicodedata

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)
```

### Step 2.2: PII De-identification

Remove personal identifiers from case law and contracts. Use regex patterns adapted to Vietnamese naming conventions.

```python
import re

VI_NAME_PATTERN   = r'\b(Nguyễn|Trần|Lê|Phạm|Hoàng|Huỳnh|Phan|Vũ|Võ|Đặng|Bùi|Đỗ|Hồ|Ngô|Dương)\s+[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠẶ][a-zàáâãèéêìíòóôõùúăđĩũơưạặ]+(?:\s+[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠẶ][a-zàáâãèéêìíòóôõùúăđĩũơưạặ]+)?'
CCCD_PATTERN      = r'\b\d{9,12}\b'
PHONE_PATTERN     = r'\b(0|\+84)[3-9]\d{8}\b'

def deidentify(text: str) -> str:
    text = re.sub(VI_NAME_PATTERN, "[HO_TEN]", text)
    text = re.sub(CCCD_PATTERN, "[SO_GIAY_TO]", text)
    text = re.sub(PHONE_PATTERN, "[SO_DIEN_THOAI]", text)
    return text
```

### Step 2.3: Legal Citation Normalization

Standardize references to laws, articles, and clauses to a canonical form so the model learns consistent citation syntax.

```python
# Input:  "điều 123, khoản 2 bộ luật dân sự"
# Output: "Điều 123 Khoản 2 Bộ luật Dân sự 2015"

# Use regex to detect patterns like:
# "Điều X", "Khoản Y", "Điểm Z", "BLDS", "BLHS", "Nghị định XX/YYYY/NĐ-CP"
```

### Step 2.4: Legal-Structure-Aware Chunking for RAG

Do not chunk by fixed token count. Chunk by legal structural units (Điều). Each chunk must carry full provenance metadata.

```python
def chunk_by_dieu(document: str, doc_metadata: dict) -> list[dict]:
    """
    Split a legal document at 'Điều X.' boundaries.
    Each chunk = one Article (Điều) + its sub-clauses (Khoản, Điểm).
    Overlap: prepend the first Khoản of the next Điều to avoid context loss.
    """
    chunks = []
    # Split on "Điều \d+" pattern
    dieu_pattern = re.compile(r'(Điều\s+\d+[\.:])')
    parts = dieu_pattern.split(document)
    # Reconstruct and attach metadata
    for i in range(1, len(parts), 2):
        header = parts[i]
        body   = parts[i+1] if i+1 < len(parts) else ""
        text   = (header + body).strip()
        chunks.append({
            "id":       f"{doc_metadata['van_ban']}_{header.strip()}",
            "text":     text,
            "metadata": {
                "van_ban":      doc_metadata["van_ban"],
                "nam_ban_hanh": doc_metadata["nam_ban_hanh"],
                "linh_vuc":     doc_metadata["linh_vuc"],
                "dieu":         header.strip()
            }
        })
    return chunks
```

### Step 2.5: Instruction Dataset Construction

Convert QA pairs into instruction-following format for SFT. Every output must include a legal citation and a disclaimer.

```python
INSTRUCTION_TEMPLATE = (
    "Bạn là trợ lý pháp lý chuyên về luật Việt Nam. "
    "Hãy trả lời câu hỏi dựa trên các quy định pháp luật hiện hành, "
    "trích dẫn rõ điều khoản và văn bản pháp lý liên quan."
)

def build_sft_sample(question: str, context: str, answer: str, citation: str) -> dict:
    return {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": f"Câu hỏi: {question}\n\nNguồn tham chiếu:\n{context}",
        "output": (
            f"{answer}\n\n"
            f"Căn cứ pháp lý: {citation}\n\n"
            f"Lưu ý: Nội dung trên chỉ mang tính chất tham khảo, "
            f"không phải tư vấn pháp lý chính thức. "
            f"Hãy tham khảo luật sư có chuyên môn để được tư vấn cụ thể."
        )
    }
```

### Step 2.6: Preference Dataset Construction (for DPO)

Generate chosen/rejected pairs from `ViBidLQA_v1`. The "chosen" answer includes a valid citation; the "rejected" answer is the same content without citation or with a hallucinated article number.

```python
def build_dpo_sample(prompt: str, good_answer: str, bad_answer: str) -> dict:
    return {
        "prompt":   prompt,
        "chosen":   good_answer,   # With correct citation
        "rejected": bad_answer     # Without citation or with fabricated number
    }
```

### Step 2.7: Quality Filtering

Remove samples that fail length or perplexity thresholds.

```python
MIN_CHARS = 100
MAX_CHARS = 4096

def passes_quality_filter(sample: dict) -> bool:
    text = sample.get("output", sample.get("text", ""))
    if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
        return False
    return True
```

### Output Files

| File                   | Path                                 | Description                           |
| ---------------------- | ------------------------------------ | ------------------------------------- |
| `cpt_corpus.jsonl`   | `PROCESSED_DIR/cpt_corpus.jsonl`   | One document per line for CPT         |
| `sft_train.jsonl`    | `PROCESSED_DIR/sft_train.jsonl`    | Instruction samples for SFT           |
| `sft_val.jsonl`      | `PROCESSED_DIR/sft_val.jsonl`      | Validation split (10%)                |
| `dpo_train.jsonl`    | `PROCESSED_DIR/dpo_train.jsonl`    | Preference pairs for DPO              |
| `rag_passages.jsonl` | `PROCESSED_DIR/rag_passages.jsonl` | Chunked passages with metadata        |
| `eval_gold.jsonl`    | `PROCESSED_DIR/eval_gold.jsonl`    | Gold evaluation set                   |
| `data_stats.json`    | `LOG_DIR/data_stats.json`          | Token counts, sample counts per split |

### Acceptance Criteria

- Unicode NFC normalization applied to 100% of corpus
- PII patterns not detectable in `sft_train.jsonl`
- Every SFT output contains at least one citation and the disclaimer string
- Every RAG chunk has `van_ban`, `nam_ban_hanh`, `linh_vuc`, `dieu` metadata fields
- `data_stats.json` logged and reviewed

---

## Stage 3: Continued Pre-Training (CPT)

**Duration:** Week 2
**Notebook:** `notebooks/03_cpt_training.ipynb`
**When to run:** Only if zero-shot LegalBench VN score is below 65%. Otherwise skip to Stage 4.

### Objective

Adapt the base model's token distribution toward Vietnamese legal vocabulary. After CPT, the model should fluently predict legal terminology, article references, and decree structures without fine-tuning.

### Model Selection Decision Tree

```
Is the zero-shot VN legal score >= 65%?
    YES --> Skip CPT, go directly to Stage 4 (SFT)
    NO  --> Run CPT for 1–2 epochs on cpt_corpus.jsonl
```

### Recommended Base Models (in priority order)

| Rank | Model                                       | Parameters | Rationale                                                     |
| ---- | ------------------------------------------- | ---------- | ------------------------------------------------------------- |
| 1    | `Qwen/Qwen2.5-7B-Instruct`                | 7B         | Best Vietnamese comprehension, instruction-tuned base         |
| 2    | `Qwen/Qwen2.5-3B-Instruct`                | 3B         | Faster, fits T4 with less quantization overhead               |
| 3    | `thangvip/qwen3-4b-vietnamese-legal-grpo` | 4B         | Already GRPO-tuned on VN legal QA — preferred starting point |
| 4    | `vilm/vistral-7b-chat`                    | 7B         | Native Vietnamese, good alternative                           |
| 5    | `VinAIResearch/PhoGPT-7B5-Instruct`       | 7.5B       | Strongest native VN, high VRAM requirement                    |

### CPT Training Configuration

```python
from transformers import TrainingArguments
from trl import SFTTrainer

cpt_args = TrainingArguments(
    output_dir           = CHECKPOINT_DIR + "/cpt",
    num_train_epochs     = 2,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 16,    # Effective batch size = 16
    learning_rate        = 2e-5,
    lr_scheduler_type    = "cosine",
    warmup_ratio         = 0.05,
    fp16                 = True,
    gradient_checkpointing = True,
    save_strategy        = "steps",
    save_steps           = 200,
    logging_steps        = 50,
    report_to            = "wandb",      # or "none" if no W&B token
    run_name             = "cpt-vn-legal-v1",
)
```

### 4-bit Quantization Config (mandatory for T4)

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16,
)
```

### LoRA Config for CPT

```python
from peft import LoraConfig

cpt_lora_config = LoraConfig(
    r              = 32,
    lora_alpha     = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout   = 0.05,
    bias           = "none",
    task_type      = "CAUSAL_LM",
)
```

### Acceptance Criteria

- Training loss decreases monotonically over first 500 steps
- Perplexity on a held-out legal paragraph decreases compared to base model
- Checkpoint saved at `CHECKPOINT_DIR/cpt/`

---

## Stage 4: Supervised Fine-Tuning (SFT)

**Duration:** Week 2–3
**Notebook:** `notebooks/04_sft_training.ipynb`

### Objective

Teach the model to follow legal instruction format: receive a legal question, reason through applicable law, cite specific articles, and append the mandatory disclaimer.

### Training Configuration

```python
from trl import SFTTrainer, SFTConfig

sft_args = SFTConfig(
    output_dir                  = CHECKPOINT_DIR + "/sft",
    num_train_epochs            = 3,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    learning_rate               = 2e-4,
    lr_scheduler_type           = "cosine",
    warmup_ratio                = 0.05,
    fp16                        = True,
    gradient_checkpointing      = True,
    max_seq_length              = 2048,
    dataset_text_field          = "text",
    save_strategy               = "steps",
    save_steps                  = 100,
    eval_strategy               = "steps",
    eval_steps                  = 100,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    report_to                   = "wandb",
    run_name                    = "sft-vn-legal-v1",
)
```

### LoRA Config for SFT

```python
sft_lora_config = LoraConfig(
    r              = 16,
    lora_alpha     = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout   = 0.05,
    bias           = "none",
    task_type      = "CAUSAL_LM",
)
```

### Chat Template

Apply the following prompt template consistently across all samples:

```python
def format_prompt(sample: dict) -> str:
    return (
        f"<|im_start|>system\n{sample['instruction']}<|im_end|>\n"
        f"<|im_start|>user\n{sample['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{sample['output']}<|im_end|>"
    )
```

### Acceptance Criteria

- Eval loss decreases from epoch 1 to epoch 3
- Generated outputs contain citation pattern (`Điều X, Khoản Y`) on at least 80% of legal questions in validation set
- Generated outputs contain disclaimer string on 100% of samples
- Checkpoint saved at `CHECKPOINT_DIR/sft/`

---

## Stage 5: Preference Optimization (DPO)

**Duration:** Week 3
**Notebook:** `notebooks/05_dpo_training.ipynb`

### Objective

Improve output quality beyond what SFT can achieve by teaching the model to prefer correctly-cited, well-reasoned answers over vague or hallucinated ones.

### Algorithm Selection

Use DPO (Direct Preference Optimization) as the default. DPO requires no reward model and is stable on T4 GPU with small batch sizes.

```
SFT score >= threshold AND citation recall >= 70%
    --> Use DPO (default)
SFT score low OR training unstable
    --> Try ORPO (combined SFT+preference in one pass)
```

### DPO Training Configuration

```python
from trl import DPOTrainer, DPOConfig

dpo_args = DPOConfig(
    output_dir                  = CHECKPOINT_DIR + "/dpo",
    num_train_epochs            = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    learning_rate               = 5e-5,
    beta                        = 0.1,       # KL regularization coefficient
    fp16                        = True,
    gradient_checkpointing      = True,
    save_strategy               = "epoch",
    report_to                   = "wandb",
    run_name                    = "dpo-vn-legal-v1",
)
```

### Preference Data Construction Rules

- **Chosen:** Answer with correct article number, correct law name, and disclaimer
- **Rejected:** Same semantic content but (a) missing citation, or (b) containing a plausible but incorrect article number, or (c) missing disclaimer

### Acceptance Criteria

- DPO loss is positive and decreasing
- Citation recall on validation set improves by at least 5 percentage points over SFT checkpoint
- Hallucination rate (wrong article number) reduces compared to SFT checkpoint

---

## Stage 6: Retrieval-Augmented Generation (RAG)

**Duration:** Week 3–4
**Notebook:** `notebooks/06_rag_pipeline.ipynb`

### Objective

Ground model responses in retrieved, up-to-date legal passages. This is non-negotiable for Vietnamese law because statutes change after every National Assembly session.

### Architecture

```
User Query
    |
    v
Query Encoder (BGE-M3 multilingual)
    |
    v
Dense Retriever (FAISS IndexFlatIP)  +  Sparse Retriever (BM25)
    |                                         |
    +------------------+-----------------------+
                       |
                  Hybrid Fusion (RRF — Reciprocal Rank Fusion)
                       |
                  Re-Ranker (BGE-Reranker-v2-m3)
                       |
                  Top-K Passages (K=5) with metadata filter
                       |
                  LLM (SFT/DPO checkpoint)
                       |
                  Response + Citations
```

### Dense Index Construction

```python
from sentence_transformers import SentenceTransformer
import faiss, json, numpy as np

encoder = SentenceTransformer("BAAI/bge-m3")

passages = [json.loads(l) for l in open(f"{PROCESSED_DIR}/rag_passages.jsonl")]
texts    = [p["text"] for p in passages]

embeddings = encoder.encode(texts, batch_size=64, show_progress_bar=True,
                             normalize_embeddings=True)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype(np.float32))

faiss.write_index(index, f"{MODEL_DIR}/faiss_legal.index")
json.dump(passages, open(f"{MODEL_DIR}/passages_meta.json", "w"), ensure_ascii=False)
```

### Metadata Filtering

Before returning passages to the LLM, filter by `linh_vuc` (legal domain) and `nam_ban_hanh` (year). Only return passages from laws currently in force.

```python
CURRENT_LAWS = {
    "BLDS": 2015, "BLHS": 2015, "BLLĐộng": 2019, "BLTTDs": 2015
}

def is_in_force(passage_meta: dict) -> bool:
    van_ban = passage_meta.get("van_ban", "")
    nam = passage_meta.get("nam_ban_hanh", 0)
    for law, year in CURRENT_LAWS.items():
        if law in van_ban and nam >= year:
            return True
    return True  # Default: include if unknown
```

### RAG Prompt Template

```python
def build_rag_prompt(query: str, retrieved_passages: list[dict]) -> str:
    context_blocks = []
    for i, p in enumerate(retrieved_passages, 1):
        meta = p["metadata"]
        context_blocks.append(
            f"[Nguồn {i}] {meta['van_ban']} — {meta['dieu']}\n{p['text']}"
        )
    context = "\n\n".join(context_blocks)
    return (
        f"Dựa vào các điều khoản pháp luật sau:\n\n{context}\n\n"
        f"Câu hỏi: {query}\n\n"
        f"Hãy trả lời câu hỏi trên, trích dẫn rõ nguồn pháp lý được cung cấp."
    )
```

### Acceptance Criteria

- FAISS index built and saved at `MODEL_DIR/faiss_legal.index`
- Retrieval MRR@10 >= 0.70 on `phamson02/large-vi-legal-queries` validation set
- RAG pipeline returns top-5 passages with correct metadata in < 2 seconds on T4

---

## Stage 7: Evaluation

**Duration:** Week 4
**Notebook:** `notebooks/07_evaluation.ipynb`

### Objective

Validate the model against all defined KPIs before declaring it production-ready.

### Benchmark Suite

| Benchmark             | Metric                        | Minimum Pass | Dataset                              |
| --------------------- | ----------------------------- | ------------ | ------------------------------------ |
| Vietnamese Legal QA   | Exact Match / F1              | >= 80%       | `ntphuc149/ViBidLQA_v1`            |
| Citation Recall       | Recall@1                      | >= 0.75      | Custom gold set                      |
| Citation Precision    | Precision@1                   | >= 0.95      | Custom gold set                      |
| Hallucination Rate    | Error rate on article numbers | <= 2%        | 200 adversarial prompts              |
| Disclaimer Compliance | Rate of disclaimer presence   | 100%         | Any 100 samples                      |
| RAG MRR               | MRR@10                        | >= 0.70      | `phamson02/large-vi-legal-queries` |

### Citation Evaluation Script

```python
import re

def extract_citations(text: str) -> list[str]:
    pattern = r'(Điều\s+\d+[^\n,;]*(?:(?:Khoản|Điểm)\s+\d+[^\n,;]*)?)'
    return re.findall(pattern, text)

def citation_recall(generated: str, reference_citations: list[str]) -> float:
    gen_citations = set(extract_citations(generated))
    ref_citations = set(reference_citations)
    if not ref_citations:
        return 1.0
    return len(gen_citations & ref_citations) / len(ref_citations)
```

### Hallucination Probe

Generate 200 prompts asking about specific articles that either exist or do not exist. Measure the rate at which the model invents article numbers.

```python
PROBE_TEMPLATE = "Điều {article_num} Bộ luật Dân sự 2015 quy định gì?"
NONEXISTENT_ARTICLES = [999, 1001, 500, 777]  # Articles that do not exist

def run_hallucination_probe(model, tokenizer, articles: list[int]) -> float:
    hallucinations = 0
    for art in articles:
        prompt = PROBE_TEMPLATE.format(article_num=art)
        output = generate(model, tokenizer, prompt)
        # If model describes content for a nonexistent article, it hallucinated
        if f"Điều {art}" in output and "không tồn tại" not in output.lower():
            hallucinations += 1
    return hallucinations / len(articles)
```

### Decision Gate

```
ALL of the following must be true to proceed to Stage 8:
    - VN Legal QA F1    >= 80%
    - Citation Recall   >= 0.75
    - Citation Precision >= 0.95
    - Hallucination Rate <= 2%
    - Disclaimer Rate   == 100%

If any criterion fails:
    - Citation failures  --> Return to Stage 5 (DPO) with more preference pairs
    - Hallucination failures --> Improve RAG retrieval quality (Stage 6)
    - Low QA scores --> Return to Stage 4 (SFT) with more training data
```

---

## Stage 8: Quantization and Deployment

**Duration:** Week 4
**Notebook:** `notebooks/08_quantization_deploy.ipynb`

### Objective

Compress the trained model for efficient inference and deploy a usable demo at zero cost.

### Step 8.1: Merge LoRA Adapter into Base Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR + "/dpo")
merged_model = model.merge_and_unload()

merged_model.save_pretrained(MODEL_DIR + "/merged")
tokenizer.save_pretrained(MODEL_DIR + "/merged")
print("Merged model saved.")
```

### Step 8.2: 4-bit GGUF Quantization (for local deployment)

```bash
# Run on CPU session after merging
!git clone https://github.com/ggerganov/llama.cpp
!cd llama.cpp && pip install -r requirements.txt
!python llama.cpp/convert_hf_to_gguf.py /kaggle/working/model/merged \
    --outfile /kaggle/working/model/legal_vn_q4.gguf \
    --outtype q4_k_m
```

### Step 8.3: Push to HuggingFace Hub

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path = MODEL_DIR + "/merged",
    repo_id     = "vtgh04/legal-vn-llm-v1",
    repo_type   = "model",
)
```

### Step 8.4: Gradio Demo (optional, free on HuggingFace Spaces)

```python
import gradio as gr

DISCLAIMER = (
    "\n\n---\nLuu y: Noi dung tren chi mang tinh chat tham khao, "
    "khong phai tu van phap ly chinh thuc. "
    "Hay tham khao luat su co chuyen mon de duoc tu van cu the."
)

def legal_chat(query: str) -> str:
    passages = retrieve(query, k=5)
    prompt   = build_rag_prompt(query, passages)
    response = generate(model, tokenizer, prompt)
    return response + DISCLAIMER

gr.Interface(fn=legal_chat, inputs="text", outputs="text",
             title="Legal LLM Vietnam (Demo)").launch()
```

---

## Timeline Summary

| Week | Stage                              | Key Output                                     |
| ---- | ---------------------------------- | ---------------------------------------------- |
| 1    | Data acquisition + preprocessing   | `PROCESSED_DIR/*.jsonl`, `data_stats.json` |
| 2    | CPT (if needed) + SFT start        | CPT checkpoint, SFT checkpoint (epoch 1)       |
| 3    | SFT complete + DPO + RAG index     | DPO checkpoint, FAISS index                    |
| 4    | Evaluation + quantization + deploy | GGUF model, Gradio demo, eval report           |
| 5–9 | Iteration based on evaluation gaps | Improved checkpoints                           |

---

## Ethical and Legal Guardrails

The following rules are applied at every inference, without exception.

1. Every response must contain the Vietnamese disclaimer:*"Nội dung này chỉ mang tính chất tham khảo, không phải tư vấn pháp lý chính thức. Hãy tham khảo luật sư có chuyên môn."*
2. The model must not advise on specific litigation strategy, draft court filings, or impersonate a licensed attorney.
3. All citations must link to a retrieved source passage. The model must not generate article numbers without a corresponding retrieved passage in the context.
4. Jurisdiction must be stated explicitly in every legal interpretation. If the applicable jurisdiction is ambiguous, the model must ask the user to clarify before proceeding.
5. No personally identifiable information from training data may appear in model outputs.

---

## Agent Execution Rules

1. Read `KAGGLE_ENV.md` before writing any code.
2. Read `SKILL.md` before planning any pipeline decision.
3. Every code file produced must match the stage and file it belongs to, as defined in `FOLDER_STRUCTURE.md`.
4. Agents write code only. Agents do not execute cells.
5. All file paths in generated code must use the constants defined in Stage 0.
6. No hyperparameter may be hardcoded in `.py` files. All values come from `configs/*.yaml`.
