"""
Build 04-sft-training-v2.ipynb — FIXED version.

ROOT CAUSE FIX:
  Original notebook merged CPT LoRA into 4-bit quantized weights (lossy),
  then applied a NEW LoRA on top. This caused corrupted base weights
  and gibberish inference output (eval accuracy 2%).

FIX:
  Strategy A (default): Skip CPT merge entirely. Train SFT from base model.
    - SFT data already contains legal content — CPT knowledge is redundant.
    - Completely avoids 4-bit merge corruption.

  Strategy B (optional): Load CPT adapter as PEFT, continue training its
    LoRA weights on SFT data WITHOUT merging into 4-bit base.
    - Set USE_CPT_ADAPTER = True in CONFIG.

Run:  python scripts/make_notebook04_v2.py
"""
import json, os

NB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "notebooks", "04-sft-training-v2.ipynb"
)


def md(src):
    return {"cell_type": "markdown", "source": src, "metadata": {}}


def code(src):
    return {"cell_type": "code", "source": src, "metadata": {"trusted": True},
            "outputs": [], "execution_count": None}


# ──────────────────────────────────────────────────────────────
# Markdown header
# ──────────────────────────────────────────────────────────────
HEADER = (
    "# Stage 4 — Supervised Fine-Tuning (SFT) — v2 FIXED\n\n"
    "**Thay đổi so với v1:**\n"
    "1. ❌ KHÔNG merge CPT LoRA vào model 4-bit (gây corruption → gibberish output)\n"
    "2. ✅ Hai chiến lược:\n"
    "   - **A (mặc định):** SFT từ base model — đơn giản, ổn định\n"
    "   - **B (tùy chọn):** Load CPT adapter → tiếp tục train LoRA trên SFT data\n"
    "3. ✅ Dùng Kaggle Secrets cho HF_TOKEN (không hardcode)\n"
    "4. ✅ Giảm epochs 3→2, thêm eval monitoring\n\n"
    "**Chạy:** Save Version → Save & Run All"
)

# ──────────────────────────────────────────────────────────────
CELL0 = '%pip install -q "unsloth[kaggle-new]" wandb huggingface_hub pyvi'

# ──────────────────────────────────────────────────────────────
CELL1 = r'''# Cell 1 — Paths + Config
import os, sys, gc, json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Force single GPU
import torch

print(f"Python  : {sys.version}")
print(f"PyTorch : {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ---- HF Token from Kaggle Secrets (KHÔNG hardcode) ----
HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    print(f"HF_TOKEN: {HF_TOKEN[:8]}..." if HF_TOKEN else "HF_TOKEN: [EMPTY]")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    print(f"HF_TOKEN from env: {'OK' if HF_TOKEN else 'NOT SET'}")

# ---- Dataset paths ----
STAGE2_DATASET = "/kaggle/input/datasets/giahuyvotran/egal-llm-stage2-processed"
SFT_FILES = {
    "sft_train":      f"{STAGE2_DATASET}/data_processed/sft/sft_train.jsonl",
    "vn_legal_qa":    f"{STAGE2_DATASET}/data_processed/sft/sft_vn_legal_qa.jsonl",
    "vn_legal_chat":  f"{STAGE2_DATASET}/data_processed/sft/sft_vn_legal_chat.jsonl",
}
EVAL_GOLD = f"{STAGE2_DATASET}/data_processed/eval/eval_gold.jsonl"

OUTPUT_BASE  = "/kaggle/working"
MODEL_DIR    = f"{OUTPUT_BASE}/sft_model"
ADAPTER_DIR  = f"{OUTPUT_BASE}/sft_adapter"
LOG_DIR      = f"{OUTPUT_BASE}/logs"
for d in [MODEL_DIR, ADAPTER_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp luật Việt Nam chuyên nghiệp. "
    "Hãy trả lời câu hỏi dựa trên quy định pháp luật hiện hành. "
    "Luôn trích dẫn điều luật cụ thể khi có thể."
)

CONFIG = {
    # ──── MODEL ────
    "base_model_id":         "unsloth/Qwen2.5-7B-bnb-4bit",
    "cpt_adapter_id":        "vtgh1602/legal-llm-cpt-qwen25-7b-adapter",
    "max_seq_length":        2048,

    # ──── STRATEGY ────
    # False = Strategy A: SFT from base model (recommended, avoids 4-bit merge bug)
    # True  = Strategy B: Load CPT adapter and continue training its LoRA
    "use_cpt_adapter":       False,

    # ──── LORA (only used when use_cpt_adapter=False) ────
    "lora_r":                32,
    "lora_alpha":            64,
    "lora_dropout":          0.0,     # 0.0 = fast patching in Unsloth
    "lora_target_modules":   ["q_proj","k_proj","v_proj","o_proj",
                              "gate_proj","up_proj","down_proj"],

    # ──── DATA ────
    "max_train_records":     25_000,
    "min_chars":             50,

    # ──── TRAINING ────
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "num_train_epochs":      2,       # Reduced from 3 to prevent overfitting
    "max_steps":             -1,
    "warmup_steps":          100,
    "learning_rate":         2e-4,    # Slightly higher for from-scratch SFT
    "lr_scheduler_type":     "cosine",
    "weight_decay":          0.01,
    "seed":                  42,
    "logging_steps":         25,
    "save_steps":            200,

    # ──── OUTPUT ────
    "output_dir":            MODEL_DIR,
    "hf_model_id":           "vtgh1602/legal-llm-sft-v2-qwen25-7b",
    "hf_checkpoint_repo":    "vtgh1602/legal-llm-sft-v2-checkpoints",
    "push_to_hub":           True,
}

for k, v in SFT_FILES.items():
    print(f"  {k:<15}: {'OK' if os.path.exists(v) else 'NOT FOUND'}")
print(f"  eval_gold      : {'OK' if os.path.exists(EVAL_GOLD) else 'NOT FOUND'}")
print(f"\nStrategy: {'B (CPT adapter → continue SFT)' if CONFIG['use_cpt_adapter'] else 'A (base model → fresh SFT)'}")
print("Config loaded.")
'''

# ──────────────────────────────────────────────────────────────
CELL2 = r'''# Cell 2 — Load & Format SFT Data (ChatML Template)
import json as _json, random, os
from datasets import Dataset

random.seed(CONFIG["seed"])

def to_chatml(instruction, inp, output):
    user_msg = (instruction + "\n\n" + inp).strip() if inp else instruction.strip()
    return (
        "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n"
        "<|im_start|>user\n" + user_msg + "<|im_end|>\n"
        "<|im_start|>assistant\n" + output + "<|im_end|>"
    )

def load_jsonl(path, label):
    records = []
    if not os.path.exists(path):
        print(f"  [SKIP] {label}: not found"); return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r   = _json.loads(line)
                ins = r.get("instruction", r.get("prompt", ""))
                inp = r.get("input",       r.get("context", ""))
                out = r.get("output",      r.get("response", r.get("answer", "")))
                if ins and out and len(out) >= CONFIG["min_chars"]:
                    records.append(to_chatml(ins, inp, out))
            except: continue
    print(f"  {label:<15}: {len(records):>6,} records")
    return records

print("Loading SFT datasets...")
all_texts = []
for label, path in SFT_FILES.items():
    all_texts.extend(load_jsonl(path, label))

print(f"  Total (raw)  : {len(all_texts):,}")
all_texts = list(set(all_texts))
print(f"  After dedup  : {len(all_texts):,}")
random.shuffle(all_texts)

# Split train / val (95/5)
n_val = max(500, int(len(all_texts) * 0.05))
val_texts   = all_texts[:n_val]
train_texts = all_texts[n_val:n_val + CONFIG["max_train_records"]]

print(f"  Train        : {len(train_texts):,}")
print(f"  Val          : {len(val_texts):,}")
print(f"  Avg chars    : {sum(len(t) for t in train_texts)//len(train_texts):,}")

train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset   = Dataset.from_dict({"text": val_texts})
print(f"  Train dataset: {train_dataset}")
print(f"  Val dataset  : {val_dataset}")
print("\nSample[0]:\n" + train_dataset[0]["text"][:400])
'''

# ──────────────────────────────────────────────────────────────
CELL3 = r'''# Cell 3 — Load Base Model (4-bit)
import gc, torch
gc.collect(); torch.cuda.empty_cache()
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CONFIG["base_model_id"],
    max_seq_length=CONFIG["max_seq_length"],
    load_in_4bit=True,
    dtype=None,
)
print(f"Base model loaded: {CONFIG['base_model_id']}")
print(f"Vocab : {len(tokenizer):,}")
print(f"VRAM  : {torch.cuda.memory_allocated()/1e9:.2f} GB")
'''

# ──────────────────────────────────────────────────────────────
CELL4_FIXED = r'''# Cell 4 — [FIXED] Apply LoRA adapter
# ============================================================
# v1 BUG: merge_and_unload() on 4-bit model caused weight corruption.
#   → Inference produced garbage: "s fairкров .yy..."
#   → Eval accuracy: 2%
#
# v2 FIX:
#   Strategy A (default): Fresh LoRA on base model — no CPT merge at all
#   Strategy B (optional): Load CPT adapter WITHOUT merging, train same LoRA
# ============================================================
from unsloth import FastLanguageModel

if CONFIG["use_cpt_adapter"]:
    # ── Strategy B: Load CPT adapter, continue training its LoRA ──
    from peft import PeftModel
    print(f"Loading CPT adapter: {CONFIG['cpt_adapter_id']}")
    print("  (NOT merging — keeping as trainable LoRA)")

    model = PeftModel.from_pretrained(
        model, CONFIG["cpt_adapter_id"],
        token=HF_TOKEN if HF_TOKEN else None,
        is_trainable=True,     # Keep LoRA weights trainable for SFT
    )

    # Enable gradient checkpointing for memory efficiency
    model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    _train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _total = sum(p.numel() for p in model.parameters())
    print(f"CPT LoRA loaded (NOT merged)!")
    print(f"Trainable: {_train/1e6:.1f}M / {_total/1e6:.1f}M ({100*_train/_total:.2f}%)")
    print(f"LoRA rank: 64 (from CPT adapter)")

else:
    # ── Strategy A (DEFAULT): Fresh LoRA on clean base model ──
    print("Strategy A: Applying fresh LoRA on base model (no CPT)")

    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["lora_target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=CONFIG["seed"],
        use_rslora=True,
    )

    _train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {_train/1e6:.1f}M / {_total/1e6:.1f}M ({100*_train/_total:.2f}%)")
    print(f"LoRA rank: {CONFIG['lora_r']}")

print(f"VRAM  : {torch.cuda.memory_allocated()/1e9:.2f} GB")
'''

# ──────────────────────────────────────────────────────────────
CELL5_RESUME = r'''# Cell 5 — Auto-Resume: Download checkpoint from HF Hub
from huggingface_hub import HfApi, snapshot_download
import shutil

CHECKPOINT_REPO = CONFIG["hf_checkpoint_repo"]

def get_latest_step(repo_id, token):
    try:
        path = HfApi().hf_hub_download(repo_id=repo_id, filename="latest_step.txt",
                                        repo_type="model", token=token)
        with open(path) as f: return int(f.read().strip())
    except: return None

def download_ckpt(repo_id, step, base, token):
    local = os.path.join(base, f"checkpoint-{step}")
    os.makedirs(local, exist_ok=True)
    print(f"  Downloading checkpoint-{step} from Hub...")
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=local,
                      allow_patterns=[f"checkpoint-{step}/*"], token=token,
                      ignore_patterns=["*.msgpack","flax_model*"])
    nested = os.path.join(local, f"checkpoint-{step}")
    if os.path.isdir(nested):
        for item in os.listdir(nested):
            shutil.move(os.path.join(nested, item), os.path.join(local, item))
        os.rmdir(nested)
    print(f"  Saved to: {local}")
    for fn in sorted(os.listdir(local)):
        sz = os.path.getsize(os.path.join(local, fn)) / 1024**2
        print(f"    {fn:<40} {sz:>8.2f} MB")
    return local

resume_checkpoint = None
if HF_TOKEN:
    step = get_latest_step(CHECKPOINT_REPO, HF_TOKEN)
    if step:
        print(f"Found checkpoint step {step} on Hub — Resuming!")
        resume_checkpoint = download_ckpt(CHECKPOINT_REPO, step, CONFIG["output_dir"], HF_TOKEN)
    else:
        print("No checkpoint on Hub → Training from scratch")
else:
    print("HF_TOKEN not set → Training from scratch")
print(f"resume_checkpoint = {resume_checkpoint}")
'''

# ──────────────────────────────────────────────────────────────
CELL6_TRAIN = r'''# Cell 6 — SFT Training (with eval loss monitoring)
import gc, datetime, os, torch
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer as Trainer, UnslothTrainingArguments as TrainConfig
from transformers import TrainerCallback
from huggingface_hub import HfApi

# ── HOTFIX: Triton fp32/fp64 type mismatch on T4/fp16 (unsloth_zoo 2026.4.x) ──
# Unsloth's fused CE loss uses @torch.compile on `accumulate_chunk`.
# The compiled Triton kernel initializes _tmp17 as fp32 but a Python float
# (fp64) reassigns it inside the loop → Triton sm_75 (T4) rejects this.
#
# suppress_errors=True does NOT fix this because the error happens in a
# Triton subprocess worker, not in the main dynamo eval frame.
#
# REAL FIX: Completely disable torch.compile / TorchDynamo.
# Training runs in eager mode (~5-10% slower) but avoids all Triton bugs.
import torch._dynamo
torch._dynamo.config.disable = True
torch._dynamo.reset()
print("[FIX] torch._dynamo DISABLED — eager mode active (Triton fp32/fp64 workaround)")

gc.collect(); torch.cuda.empty_cache(); model.train()
IS_BF16 = is_bfloat16_supported()
print(f"BF16: {IS_BF16} | Token: {'OK' if HF_TOKEN else 'NOT SET'}")

class PushCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if not HF_TOKEN: return control
        step = state.global_step
        ckpt = os.path.join(args.output_dir, f"checkpoint-{step}")
        try:
            model.push_to_hub(CONFIG["hf_model_id"], token=HF_TOKEN)
            tokenizer.push_to_hub(CONFIG["hf_model_id"], token=HF_TOKEN)
            print(f"[UP] Adapter step {step}")
        except Exception as e:
            print(f"[WARN] adapter push: {e}")
        if os.path.isdir(ckpt):
            try:
                api = HfApi()
                api.create_repo(CONFIG["hf_checkpoint_repo"], repo_type="model",
                                exist_ok=True, token=HF_TOKEN)
                api.upload_folder(folder_path=ckpt, repo_id=CONFIG["hf_checkpoint_repo"],
                                  path_in_repo=f"checkpoint-{step}", repo_type="model",
                                  token=HF_TOKEN)
                api.upload_file(path_or_fileobj=str(step).encode(),
                                path_in_repo="latest_step.txt",
                                repo_id=CONFIG["hf_checkpoint_repo"],
                                repo_type="model", token=HF_TOKEN)
                print(f"[OK] Full checkpoint-{step} → Hub")
                # Cleanup old checkpoints from Hub (keep latest only)
                try:
                    tree = [e.path for e in api.list_repo_tree(
                                CONFIG["hf_checkpoint_repo"], repo_type="model", token=HF_TOKEN)
                            if e.path.startswith("checkpoint-") and "/" not in e.path
                            and not e.path.endswith(f"checkpoint-{step}")]
                    for o in sorted(tree)[:-1]:
                        api.delete_folder(path_in_repo=o, repo_id=CONFIG["hf_checkpoint_repo"],
                                          repo_type="model", token=HF_TOKEN)
                        print(f"[DEL] {o}")
                except: pass
            except Exception as e:
                print(f"[WARN] ckpt push: {e}")
        return control

training_args = TrainConfig(
    output_dir=CONFIG["output_dir"],
    run_name=f"sft-v2-{datetime.datetime.now().strftime('%m%d-%H%M')}",
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    gradient_checkpointing=True,
    optim="adamw_8bit",
    learning_rate=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
    lr_scheduler_type=CONFIG["lr_scheduler_type"],
    warmup_steps=CONFIG["warmup_steps"],
    num_train_epochs=CONFIG["num_train_epochs"],
    max_steps=CONFIG["max_steps"],
    bf16=IS_BF16, fp16=not IS_BF16,
    logging_steps=CONFIG["logging_steps"],
    save_steps=CONFIG["save_steps"],
    save_total_limit=2,
    # ── Eval monitoring (NEW in v2) ──
    eval_strategy="steps",
    eval_steps=CONFIG["save_steps"],     # Eval at same frequency as save
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    dataset_text_field="text",
    max_seq_length=CONFIG["max_seq_length"],
    packing=False,   # SFT: each conversation must be complete
    seed=CONFIG["seed"],
    dataloader_num_workers=2,
)

trainer = Trainer(
    model=model, tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,    # NEW: eval monitoring
    args=training_args,
    callbacks=[PushCallback()],
)

_eff = CONFIG["per_device_train_batch_size"] * CONFIG["gradient_accumulation_steps"]
_spe = len(train_dataset) // _eff
print("=" * 60); print("SFT v2 TRAINING SUMMARY"); print("=" * 60)
print(f"  Strategy   : {'B (CPT adapter)' if CONFIG['use_cpt_adapter'] else 'A (fresh LoRA)'}")
print(f"  Train      : {len(train_dataset):,}")
print(f"  Val        : {len(val_dataset):,}")
print(f"  Steps/ep   : {_spe:,}")
print(f"  Epochs     : {CONFIG['num_train_epochs']}")
print(f"  LR         : {CONFIG['learning_rate']}")
print(f"  Eval every : {CONFIG['save_steps']} steps")
print(f"  Resume     : {resume_checkpoint or 'scratch'}")
print("=" * 60); print("Starting SFT v2...")

train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
print("=" * 60); print("SFT v2 COMPLETE"); print("=" * 60)
print(f"  Steps : {train_result.global_step}")
print(f"  Loss  : {train_result.training_loss:.4f}")
print(f"  Time  : {train_result.metrics.get('train_runtime',0)/60:.1f} min")
'''

# ──────────────────────────────────────────────────────────────
CELL7_SAVE = r'''# Cell 7 — Save Adapter + Report
import os, json as _json

model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"Adapter saved → {ADAPTER_DIR}")
for fn in sorted(os.listdir(ADAPTER_DIR)):
    sz = os.path.getsize(os.path.join(ADAPTER_DIR, fn)) / 1024**2
    print(f"  {fn:<42} {sz:>8.2f} MB")

rpt = {
    "stage":    "04_sft_v2",
    "strategy": "B_cpt_adapter" if CONFIG["use_cpt_adapter"] else "A_fresh_lora",
    "base":     CONFIG["base_model_id"],
    "lora_r":   CONFIG["lora_r"] if not CONFIG["use_cpt_adapter"] else 64,
    "records":  len(train_dataset),
    "val_size": len(val_dataset),
    "loss":     train_result.training_loss,
    "steps":    train_result.global_step,
    "runtime_min": train_result.metrics.get("train_runtime", 0) / 60,
}
with open(f"{LOG_DIR}/stage04_sft_v2_report.json", "w") as f:
    _json.dump(rpt, f, indent=2)
print(f"Report → {LOG_DIR}/stage04_sft_v2_report.json")

if CONFIG["push_to_hub"] and HF_TOKEN:
    print(f"\nPushing final adapter → {CONFIG['hf_model_id']}")
    model.push_to_hub(CONFIG["hf_model_id"], token=HF_TOKEN)
    tokenizer.push_to_hub(CONFIG["hf_model_id"], token=HF_TOKEN)
    print("[OK] Pushed!")
'''

# ──────────────────────────────────────────────────────────────
CELL8_INFERENCE = r'''# Cell 8 — Inference Test
from unsloth import FastLanguageModel
import torch

FastLanguageModel.for_inference(model)
EOS_ID = tokenizer.convert_tokens_to_ids("<|im_end|>")

def ask(question, context=""):
    user_msg = question + ("\n\nNguồn tham chiếu:\n" + context if context else "")
    prompt = (
        "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n"
        "<|im_start|>user\n" + user_msg + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inp = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=CONFIG["max_seq_length"]).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=300, temperature=0.3,
            top_p=0.9, repetition_penalty=1.1, do_sample=True,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=EOS_ID,
        )
    generated = tokenizer.decode(out[0][inp["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
    return generated.split("<|im_end|>")[0].strip()

TESTS = [
    "Điều kiện kết hôn theo pháp luật Việt Nam là gì?",
    "Thời hạn tạm giam trong vụ án hình sự là bao lâu?",
    "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động không?",
]
print("=" * 60); print("SFT v2 INFERENCE TEST"); print("=" * 60)
for i, q in enumerate(TESTS, 1):
    ans = ask(q)
    print(f"\n[Q{i}] {q}")
    print(f"[A]  {ans[:500]}")
    print("-" * 60)

# Sanity check: is output coherent Vietnamese?
test_ans = ask(TESTS[0])
is_coherent = (
    len(test_ans) > 50
    and any(w in test_ans.lower() for w in ["điều", "luật", "quy định", "khoản", "theo"])
    and "кров" not in test_ans   # Gibberish marker from v1 bug
)
print(f"\nCoherence check: {'PASS ✓' if is_coherent else 'FAIL ✗ — model may still be broken'}")
'''

# ──────────────────────────────────────────────────────────────
CELL9_EVAL = r'''# Cell 9 — Evaluation on eval_gold.jsonl
import json as _json, os, torch

if not os.path.exists(EVAL_GOLD):
    print(f"[SKIP] {EVAL_GOLD} not found")
else:
    samples = []
    with open(EVAL_GOLD, "r", encoding="utf-8") as f:
        for line in f:
            try: samples.append(_json.loads(line))
            except: pass

    N = 100
    correct = 0
    has_citation = 0
    has_disclaimer = 0
    print(f"Evaluating {N} samples...")

    for s in samples[:N]:
        pred = ask(s.get("question",""), s.get("context","")[:500])
        ref  = s.get("answer","").strip().lower()
        pred_lower = pred.lower()

        # Soft match: any key word from reference appears in prediction
        if any(w in pred_lower for w in ref.split()[:5] if len(w) > 3):
            correct += 1

        # Citation check: Điều X / Khoản Y pattern
        import re
        if re.search(r"(Điều|điều)\s+\d+", pred):
            has_citation += 1

        # Disclaimer check
        if "tham khảo" in pred_lower or "tư vấn" in pred_lower:
            has_disclaimer += 1

    acc = correct / N
    cite_rate = has_citation / N
    disc_rate = has_disclaimer / N

    print(f"\n  SFT v2 Evaluation Results ({N} samples)")
    print(f"  {'='*50}")
    print(f"  Accuracy (soft match) : {acc:.1%}  ({correct}/{N})")
    print(f"  Citation rate         : {cite_rate:.1%}  ({has_citation}/{N})")
    print(f"  Disclaimer rate       : {disc_rate:.1%}  ({has_disclaimer}/{N})")
    print()

    if acc >= 0.75:
        print("  🟢 EXCELLENT — Ready for Stage 5 (DPO)")
    elif acc >= 0.60:
        print("  🟡 GOOD — Ready for Stage 5 (DPO)")
    elif acc >= 0.30:
        print("  🟠 FAIR — Consider more training data or epochs")
    else:
        print("  🔴 POOR — Check model weights / training pipeline")

    # Update report
    rp = f"{LOG_DIR}/stage04_sft_v2_report.json"
    if os.path.exists(rp):
        with open(rp) as f: r = _json.load(f)
        r["eval_accuracy"] = acc
        r["eval_citation_rate"] = cite_rate
        r["eval_disclaimer_rate"] = disc_rate
        r["eval_n"] = N
        with open(rp, "w") as f: _json.dump(r, f, indent=2)

    print(f"\nAdapter: {CONFIG['hf_model_id']}")
    print("Next: Stage 5 — DPO preference alignment")
'''

# ──────────────────────────────────────────────────────────────
# Assemble notebook
# ──────────────────────────────────────────────────────────────
CELLS = [
    md(HEADER),
    code(CELL0),        # 0: Install
    code(CELL1),        # 1: Paths + Config
    code(CELL2),        # 2: Load SFT Data
    code(CELL3),        # 3: Load Base Model
    code(CELL4_FIXED),  # 4: [FIXED] Apply LoRA (no merge!)
    code(CELL5_RESUME), # 5: Auto-Resume
    code(CELL6_TRAIN),  # 6: Training (with eval)
    code(CELL7_SAVE),   # 7: Save + Report
    code(CELL8_INFERENCE),  # 8: Inference Test
    code(CELL9_EVAL),   # 9: Evaluation
]

nb = {
    "metadata": {
        "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.12"},
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "dataSources": [{"sourceType": "datasetVersion", "sourceId": 15011410,
                             "datasetId": 9608703, "databundleVersionId": 15887829}],
            "dockerImageVersionId": 31287,
            "isInternetEnabled": True, "language": "python",
            "sourceType": "notebook", "isGpuEnabled": True,
        },
    },
    "nbformat_minor": 4,
    "nbformat": 4,
    "cells": CELLS,
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

kb = os.path.getsize(NB_PATH) / 1024
print(f"[OK] {NB_PATH}")
print(f"     Size : {kb:.1f} KB | Cells: {len(CELLS)}")
print()
print("Changes from v1:")
print("  1. Cell 4: NO merge_and_unload() — avoids 4-bit corruption")
print("  2. Cell 2: Train/Val split (95/5) — monitors overfitting")
print("  3. Cell 6: eval_strategy='steps' — tracks eval_loss during training")
print("  4. Cell 1: HF_TOKEN from Kaggle Secrets (not hardcoded)")
print("  5. Config: epochs 3→2, lora_dropout 0.05→0.0, LR 1e-4→2e-4")
print("  6. Cell 8: Coherence check detects gibberish output")
print("  7. Cell 9: Citation rate + Disclaimer rate metrics added")
