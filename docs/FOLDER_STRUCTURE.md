# Quy ước tổ chức thư mục & tệp tin dự án

> **Dành cho:** Developer, AI Agent, Reviewer
> **Mục đích:** Giải thích chi tiết vai trò và quy tắc đặt file cho từng thư mục trong dự án. Đọc tài liệu này trước khi tạo bất kỳ file mới nào.

---

## Nguyên tắc tổng quát

1. **Mỗi thư mục có một mục đích duy nhất** — không lẫn lộn logic giữa các tầng.
2. **`src/` chứa code có thể tái sử dụng** — `notebooks/` chỉ để thử nghiệm, không chứa logic cốt lõi.
3. **`data/raw/` là bất biến** — không bao giờ sửa dữ liệu gốc, chỉ đọc và copy sang `processed/`.
4. **File cấu hình thuộc `configs/`** — không hardcode hyperparameter trong file `.py`.
5. **Mọi code chạy trên Kaggle** phải dùng đường dẫn `/kaggle/input/` và `/kaggle/working/`.

---

## Chi tiết từng thư mục

---

### `.agents/` — Tài liệu điều hướng cho AI Agent

Thư mục này **không chứa code**. Nó chứa tài liệu hướng dẫn để AI Agent hiểu đúng dự án trước khi thực thi bất kỳ tác vụ nào.

```
.agents/
├── KAGGLE_ENV.md       ← Quy tắc môi trường Kaggle (I/O paths, GPU limits, coding rules)
├── skills/
│   └── legal-llm-specialist/
│       └── SKILL.md   ← Mô tả nhiệm vụ, năng lực, SOP, KPI của agent
└── workflows/
    ├── data-pipeline.md   ← Từng bước xử lý dữ liệu (slash command /data-pipeline)
    ├── fine-tuning.md     ← Từng bước fine-tuning (slash command /fine-tuning)
    └── evaluation.md      ← Từng bước đánh giá model (slash command /evaluation)
```

**Quy tắc đặt file:**

- Mỗi workflow là 1 file `.md` độc lập trong `workflows/`.
- Mỗi agent skill là 1 thư mục trong `skills/<tên-agent>/` chứa `SKILL.md`.
- Không đặt file `.py` hay notebook ở đây.

---

### `configs/` — Cấu hình training & inference

Chứa toàn bộ **hyperparameter và cấu hình** dưới dạng YAML. Mọi thông số đều phải ở đây, không hardcode trong code Python.

```
configs/
├── sft_template.yaml       ← Template cho SFT — copy và đặt tên theo run
├── sft_qwen2_run001.yaml   ← Cấu hình cụ thể cho 1 lần chạy SFT
├── dpo_run001.yaml         ← Cấu hình DPO
├── qlora_7b.yaml           ← Cấu hình QLoRA cho model 7B
└── eval_config.yaml        ← Cấu hình cho bộ evaluation
```

**Quy tắc đặt file:**

- Đặt tên file theo pattern: `<stage>_<base-model>_<run-id>.yaml`Ví dụ: `sft_vistral7b_run002.yaml`
- File `*_template.yaml` là bản gốc — không sửa, chỉ copy.
- Mỗi lần chạy training tạo 1 file config riêng để có thể tái tạo lại.

---

### `data/` — Dữ liệu thô, xử lý và bên ngoài

```
data/
├── raw/           ← Dữ liệu gốc TẢI VỀ — KHÔNG chỉnh sửa, không commit lên git
│   ├── corpus_luat_vn/     ← Văn bản luật gốc (PDF, TXT, DOCX)
│   └── case_law/           ← Bản án, phán quyết toà
│
├── processed/     ← Dữ liệu đã làm sạch và sẵn sàng đưa vào model
│   ├── train.jsonl         ← Tập huấn luyện (JSONL format, 1 dòng = 1 sample)
│   ├── val.jsonl           ← Tập kiểm định
│   └── test.jsonl          ← Tập kiểm tra
│
└── external/      ← Dataset từ nguồn bên ngoài (benchmark, HuggingFace)
    ├── legalbench/         ← Benchmark LegalBench
    └── cuad/               ← CUAD contract dataset
```

**Quy tắc đặt file:**

- `raw/`: Chỉ đặt file tải trực tiếp từ nguồn (CourtListener, EUR-Lex, v.v.). Không xử lý ở đây.
- `processed/`: Kết quả output của `src/data_processing/`. Format chuẩn là **JSONL** (`{"instruction": "...", "output": "..."}`).
- `external/`: Benchmark, dataset HuggingFace đã tải về. Sắp xếp theo tên nguồn.
- Cả 3 thư mục này đều nằm trong `.gitignore` — **không push dữ liệu lên git**.

---

### `docs/` — Tài liệu kỹ thuật nội bộ

Chứa tài liệu văn bản cho người dùng và developer. Không chứa code.

```
docs/
├── PROJECT_GUIDE.md        ← Tổng quan dự án, môi trường, quy trình
├── FOLDER_STRUCTURE.md     ← Bạn đang đọc file này
├── ARCHITECTURE.md         ← Kiến trúc model, sơ đồ pipeline (sẽ bổ sung)
├── DATA_SOURCES.md         ← Danh sách nguồn dữ liệu pháp lý (sẽ bổ sung)
└── eval_report.md          ← Báo cáo đánh giá model (tự động sinh bởi script)
```

**Quy tắc đặt file:**

- Chỉ chứa file `.md` (Markdown).
- File `eval_report.md` được sinh tự động bởi `src/evaluation/generate_report.py`.
- Không đặt notebook hay script ở đây.

---

### `models/` — Model weights và checkpoints

```
models/
├── pretrained/     ← Base model tải từ HuggingFace — chỉ đọc
│   └── qwen2.5-7b/ ← Thư mục model (config.json, tokenizer, weights)
│
└── checkpoints/    ← Adapter và weights sau fine-tuning
    ├── sft_run001/ ← Checkpoint của run SFT 001
    └── dpo_run001/ ← Checkpoint sau DPO
```

**Quy tắc đặt file:**

- `pretrained/`: Tải về bằng `huggingface-cli download`. 1 thư mục = 1 model.
- `checkpoints/`: Đặt tên theo run ID, khớp với file config trong `configs/`.
- **Không commit weights lên git** — dùng HuggingFace Hub hoặc Kaggle Output để lưu trữ.

---

### `notebooks/` — Jupyter Notebooks (Kaggle)

Dùng cho EDA, prototyping và demo. Logic cốt lõi phải được refactor vào `src/` sau khi hoàn thiện.

```
notebooks/
├── 01_eda_legal_corpus.ipynb       ← Khám phá dữ liệu thô
├── 02_data_preprocessing.ipynb     ← Prototype tiền xử lý (→ sẽ refactor vào src/)
├── 03_sft_training_qlora.ipynb     ← Training chính chạy trên Kaggle
├── 04_dpo_training.ipynb           ← DPO fine-tuning
├── 05_evaluation.ipynb             ← Chạy benchmark
└── 06_rag_pipeline.ipynb           ← Thử nghiệm RAG
```

**Quy tắc đặt file:**

- Đặt tên file bắt đầu bằng số thứ tự giai đoạn: `01_`, `02_`, ...
- Tên file phải mô tả rõ nội dung: `03_sft_training_qlora.ipynb`.
- Notebook chạy trên Kaggle **phải dùng đường dẫn Kaggle** (`/kaggle/input/`, `/kaggle/working/`).
- Sau khi notebook ổn định, refactor logic vào `src/` tương ứng.

---

### `scripts/` — Script tự động hóa

Các đoạn script chạy độc lập (không phải module Python, không import từ `src/`).

```
scripts/
├── download_data.py        ← Tải dataset từ HuggingFace / CourtListener
├── merge_adapter.py        ← Merge LoRA adapter vào base model
├── push_to_hub.py          ← Push model lên HuggingFace Hub
├── convert_to_gguf.sh      ← Convert model sang GGUF (cho llama.cpp)
└── setup_kaggle.sh         ← Cài dependencies trên Kaggle (chạy 1 lần)
```

**Quy tắc đặt file:**

- Script Python độc lập: đặt trực tiếp trong `scripts/`.
- Script Bash: đuôi `.sh`.
- Không import từ `src/` — nếu cần logic chung, dùng `src/utils/`.

---

### `src/` — Source code cốt lõi (importable Python package)

Đây là **trung tâm** của dự án. Toàn bộ logic được tổ chức thành package Python có thể import.

#### `src/data_processing/` — Tiền xử lý dữ liệu

```
src/data_processing/
├── __init__.py
├── cleaner.py          ← Làm sạch văn bản, xử lý OCR errors
├── deidentify.py       ← Loại bỏ PII (tên, số CMND, địa chỉ)
├── citation_normalizer.py  ← Chuẩn hóa trích dẫn luật VN
├── deduplicator.py     ← Loại bỏ trùng lặp (MinHash / exact)
├── quality_filter.py   ← Lọc theo perplexity, độ dài
└── dataset_builder.py  ← Tổng hợp output thành JSONL cho training
```

#### `src/training/` — Huấn luyện model

```
src/training/
├── __init__.py
├── sft_trainer.py      ← Supervised Fine-Tuning (dùng TRL SFTTrainer)
├── dpo_trainer.py      ← DPO / ORPO preference optimization
├── cpt_trainer.py      ← Continued Pre-Training (Stage 1)
└── peft_utils.py       ← Helper: tạo LoRA config, load PEFT model
```

#### `src/evaluation/` — Đánh giá model

```
src/evaluation/
├── __init__.py
├── run_benchmarks.py   ← Entry point chạy tất cả benchmark
├── legalbench_eval.py  ← Chạy LegalBench
├── cuad_eval.py        ← Chạy CUAD contract evaluation
├── hallucination_eval.py ← Đo tỷ lệ hallucination
├── red_team.py         ← Automated red-teaming
└── generate_report.py  ← Sinh báo cáo → docs/eval_report.md
```

#### `src/inference/` — Phục vụ dự đoán

```
src/inference/
├── __init__.py
├── predictor.py        ← Class Predictor: load model, generate text
├── rag_pipeline.py     ← RAG: retrieve + generate với citation
└── api.py              ← FastAPI endpoint (nếu deploy local)
```

#### `src/utils/` — Tiện ích dùng chung

```
src/utils/
├── __init__.py
├── logging_utils.py    ← Cấu hình logger chuẩn
├── io_utils.py         ← Đọc/ghi JSONL, YAML, checkpoint
├── kaggle_utils.py     ← Helper cho môi trường Kaggle (paths, GPU check)
└── legal_utils.py      ← Xử lý đặc thù pháp lý (trích dẫn, disclaimer)
```

---

### `tests/` — Kiểm thử

```
tests/
├── test_cleaner.py         ← Unit test src/data_processing/cleaner.py
├── test_citation.py        ← Unit test citation_normalizer.py
├── test_predictor.py       ← Integration test predictor.py
└── conftest.py             ← Pytest fixtures dùng chung
```

**Quy tắc đặt file:**

- Tên file test: `test_<module>.py` tương ứng với module trong `src/`.
- Chạy bằng: `pytest tests/`

---

## Bảng tổng hợp nhanh

| Loại file                    | Đặt ở đâu             |
| ----------------------------- | -------------------------- |
| Hyperparameter, config        | `configs/*.yaml`         |
| Dataset gốc (chưa xử lý)  | `data/raw/`              |
| Dataset sẵn sàng train      | `data/processed/*.jsonl` |
| Benchmark ngoài              | `data/external/`         |
| Base model weights            | `models/pretrained/`     |
| LoRA adapter, checkpoint      | `models/checkpoints/`    |
| Notebook Kaggle               | `notebooks/`             |
| Script chạy độc lập       | `scripts/`               |
| Logic tiền xử lý dữ liệu | `src/data_processing/`   |
| Training loop, trainer        | `src/training/`          |
| Benchmark runner, report      | `src/evaluation/`        |
| API, RAG pipeline             | `src/inference/`         |
| Helper, logger, utils         | `src/utils/`             |
| Unit / integration test       | `tests/`                 |
| Tài liệu kỹ thuật         | `docs/`                  |
| Hướng dẫn agent AI         | `.agents/`               |

---

*Mọi ngoại lệ phải được ghi chú rõ trong PR description hoặc file README của thư mục tương ứng.*
