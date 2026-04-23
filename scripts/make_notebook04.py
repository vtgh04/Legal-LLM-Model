"""Build 04-sft-training.ipynb without nested f-string issues."""
import json, os

NB_PATH = r"c:\Users\ADMIN\Desktop\Legal-LLM-Model\notebooks\04-sft-training.ipynb"

def md(src): return {"cell_type":"markdown","source":src,"metadata":{}}
def code(src): return {"cell_type":"code","source":src,"metadata":{"trusted":True},"outputs":[],"execution_count":None}

CELL0 = """%pip install -q "unsloth[kaggle-new]" wandb huggingface_hub pyvi"""

CELL1 = '''# Cell 1 — Paths + SFT Config
import os, sys, gc, json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Force single GPU — Unsloth bugs with T4x2
import torch

print(f"PyTorch : {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

STAGE2_DATASET = "/kaggle/input/datasets/giahuyvotran/egal-llm-stage2-processed"
SFT_FILES = {
    "sft_train" : f"{STAGE2_DATASET}/data_processed/sft/sft_train.jsonl",
    "vn_legal_qa" : f"{STAGE2_DATASET}/data_processed/sft/sft_vn_legal_qa.jsonl",
    "vn_legal_chat" : f"{STAGE2_DATASET}/data_processed/sft/sft_vn_legal_chat.jsonl",
}
EVAL_GOLD = f"{STAGE2_DATASET}/data_processed/eval/eval_gold.jsonl"

OUTPUT_BASE  = "/kaggle/working"
MODEL_DIR    = f"{OUTPUT_BASE}/sft_model"
ADAPTER_DIR  = f"{OUTPUT_BASE}/sft_adapter"
LOG_DIR      = f"{OUTPUT_BASE}/logs"
for d in [MODEL_DIR, ADAPTER_DIR, LOG_DIR]: os.makedirs(d, exist_ok=True)

SYSTEM_PROMPT = (
    "Ban la tro ly phap luat Viet Nam chuyen nghiep. "
    "Hay tra loi cau hoi dua tren quy dinh phap luat hien hanh. "
    "Luon trich dan dieu luat cu the khi co the."
)

CONFIG = {
    "cpt_adapter_id":        "vtgh1602/legal-llm-cpt-qwen25-7b-adapter",
    "base_model_id":         "unsloth/Qwen2.5-7B-bnb-4bit",
    "max_seq_length":        2048,
    "lora_r":                32,
    "lora_alpha":            64,
    "lora_dropout":          0.05,
    "lora_target_modules":   ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    "max_train_records":     25_000,
    "min_chars":             50,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "num_train_epochs":      3,
    "max_steps":             -1,
    "warmup_steps":          50,
    "learning_rate":         1e-4,
    "lr_scheduler_type":     "cosine",
    "weight_decay":          0.01,
    "seed":                  42,
    "logging_steps":         25,
    "save_steps":            100,
    "output_dir":            MODEL_DIR,
    "hf_model_id":           "vtgh1602/legal-llm-sft-qwen25-7b-adapter",
    "hf_checkpoint_repo":    "vtgh1602/legal-llm-sft-checkpoints",
    "push_to_hub":           True,
}
for k, v in SFT_FILES.items():
    print(f"  {k:<15}: {'OK' if os.path.exists(v) else 'NOT FOUND'}")
print("Config loaded.")
'''

CELL2 = '''# Cell 2 — Load & Format SFT Data (ChatML Template)
import json as _json, random, os
from datasets import Dataset

random.seed(CONFIG["seed"])

def to_chatml(instruction, inp, output):
    user_msg = (instruction + "\\n\\n" + inp).strip() if inp else instruction.strip()
    return (
        "<|im_start|>system\\n" + SYSTEM_PROMPT + "<|im_end|>\\n"
        "<|im_start|>user\\n" + user_msg + "<|im_end|>\\n"
        "<|im_start|>assistant\\n" + output + "<|im_end|>"
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
train_texts = all_texts[:CONFIG["max_train_records"]]
print(f"  Sampled      : {len(train_texts):,}")
print(f"  Avg chars    : {sum(len(t) for t in train_texts)//len(train_texts):,}")
dataset = Dataset.from_dict({"text": train_texts})
print(f"  Dataset      : {dataset}")
print("\\nSample[0]:\\n" + dataset[0]["text"][:400])
'''

CELL3 = '''# Cell 3 — Load Base Model (4-bit)
import gc, torch; gc.collect(); torch.cuda.empty_cache()
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CONFIG["base_model_id"], max_seq_length=CONFIG["max_seq_length"],
    load_in_4bit=True, dtype=None,
)
print(f"Vocab : {len(tokenizer):,}")
print(f"VRAM  : {torch.cuda.memory_allocated()/1e9:.2f} GB")
'''

CELL4 = '''# Cell 4 — Merge CPT Adapter (Stage 3) into base
# CPT adapter dạy model hiểu luật → merge vào base model trước khi SFT
from peft import PeftModel

HF_TOKEN = os.environ.get("HF_TOKEN BEST", "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")
print(f"Merging: {CONFIG['cpt_adapter_id']}")
peft_model = PeftModel.from_pretrained(
    model, CONFIG["cpt_adapter_id"],
    token=HF_TOKEN if HF_TOKEN else None,
)
model = peft_model.merge_and_unload()

# Cast to float16 — merge_and_unload() can leave some weights in FP32,
# which causes dtype mismatch (Half vs float) in Unsloth's fast_lora kernels
# on T4 GPUs that use fp16 training instead of bf16.
for param in model.parameters():
    if param.data.dtype == torch.float32 and not hasattr(param, "quant_state"):
        param.data = param.data.half()

print(f"CPT merged! VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
'''

CELL5 = '''# Cell 5 — Apply SFT LoRA (rank 32, nhỏ hơn CPT rank 64)
from unsloth import FastLanguageModel

model = FastLanguageModel.get_peft_model(
    model,
    r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    target_modules=CONFIG["lora_target_modules"],
    bias="none", use_gradient_checkpointing="unsloth",
    random_state=CONFIG["seed"], use_rslora=True,
)
_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
_total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {_train/1e6:.1f}M / {_total/1e6:.1f}M ({100*_train/_total:.2f}%)")
print(f"lora_dropout=0.05 → Using standard LoRA (avoids fast_lora dtype bug on T4)")
'''

CELL6 = '''# Cell 6 — Auto-Resume: Download checkpoint từ HF Hub (nếu có)
from huggingface_hub import HfApi, snapshot_download
import shutil

HF_TOKEN = os.environ.get("HF_TOKEN BEST", "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")
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
        for item in os.listdir(nested): shutil.move(os.path.join(nested, item), os.path.join(local, item))
        os.rmdir(nested)
    print(f"  Saved to: {local}")
    for f in sorted(os.listdir(local)):
        print(f"    {f:<40} {os.path.getsize(os.path.join(local,f))/1024**2:>8.2f} MB")
    return local

resume_checkpoint = None
if HF_TOKEN:
    step = get_latest_step(CHECKPOINT_REPO, HF_TOKEN)
    if step:
        print(f"Found checkpoint step {step} on Hub — Resuming!")
        resume_checkpoint = download_ckpt(CHECKPOINT_REPO, step, CONFIG["output_dir"], HF_TOKEN)
    else:
        print("No checkpoint on Hub → Starting fresh (LAN 1)")
else:
    print("HF_TOKEN not set → Starting fresh")
print(f"resume_checkpoint = {resume_checkpoint}")
'''

CELL7 = '''# Cell 7 — SFT Training
# packing=False (khác CPT=True) vì mỗi conversation phải hoàn chỉnh
import gc, datetime, os, torch
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer as SFTTrainer, UnslothTrainingArguments as SFTConfig
from transformers import TrainerCallback
from huggingface_hub import HfApi

gc.collect(); torch.cuda.empty_cache(); model.train()
IS_BF16  = is_bfloat16_supported()
HF_TOKEN = os.environ.get("HF_TOKEN BEST", "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")
print(f"BF16: {IS_BF16} | Token: {'OK' if HF_TOKEN else 'NOT SET'}")

class PushCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if not HF_TOKEN: return control
        step = state.global_step
        ckpt = os.path.join(args.output_dir, f"checkpoint-{step}")
        # Push adapter (dung cho inference)
        try:
            model.push_to_hub(CONFIG["hf_model_id"], token=HF_TOKEN)
            tokenizer.push_to_hub(CONFIG["hf_model_id"], token=HF_TOKEN)
            print(f"[UP] Adapter step {step} → {CONFIG['hf_model_id']}")
        except Exception as e: print(f"[WARN] adapter: {e}")
        # Push full checkpoint (de resume optimizer/scheduler)
        if os.path.isdir(ckpt):
            try:
                api = HfApi()
                api.create_repo(CONFIG["hf_checkpoint_repo"], repo_type="model", exist_ok=True, token=HF_TOKEN)
                api.upload_folder(folder_path=ckpt, repo_id=CONFIG["hf_checkpoint_repo"],
                                   path_in_repo=f"checkpoint-{step}", repo_type="model", token=HF_TOKEN)
                api.upload_file(path_or_fileobj=str(step).encode(), path_in_repo="latest_step.txt",
                                repo_id=CONFIG["hf_checkpoint_repo"], repo_type="model", token=HF_TOKEN)
                print(f"[OK] Full checkpoint-{step} → Hub")
                # Xoa checkpoint cu (giu 1 gan nhat)
                try:
                    tree = [e.path for e in api.list_repo_tree(CONFIG["hf_checkpoint_repo"], repo_type="model", token=HF_TOKEN)
                            if e.path.startswith("checkpoint-") and "/" not in e.path and not e.path.endswith(f"checkpoint-{step}")]
                    for o in sorted(tree)[:-1]:
                        api.delete_folder(path_in_repo=o, repo_id=CONFIG["hf_checkpoint_repo"], repo_type="model", token=HF_TOKEN)
                        print(f"[DEL] {o}")
                except: pass
            except Exception as e: print(f"[WARN] ckpt: {e}")
        return control

training_args = SFTConfig(
    output_dir=CONFIG["output_dir"],
    run_name=f"sft-{datetime.datetime.now().strftime('%m%d-%H')}",
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    gradient_checkpointing=True,
    optim="adamw_8bit",
    learning_rate=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"],
    lr_scheduler_type=CONFIG["lr_scheduler_type"], warmup_steps=CONFIG["warmup_steps"],
    num_train_epochs=CONFIG["num_train_epochs"], max_steps=CONFIG["max_steps"],
    bf16=IS_BF16, fp16=not IS_BF16,
    logging_steps=CONFIG["logging_steps"], save_steps=CONFIG["save_steps"], save_total_limit=2,
    report_to="none",
    dataset_text_field="text", max_seq_length=CONFIG["max_seq_length"],
    packing=False,            # SFT: conversations phải nguyên vẹn
    seed=CONFIG["seed"], dataloader_num_workers=0,
)

trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset,
                      args=training_args, callbacks=[PushCallback()])

_eff = CONFIG["per_device_train_batch_size"] * CONFIG["gradient_accumulation_steps"]
_spe = len(dataset) // _eff
print("="*60); print("SFT TRAINING SUMMARY"); print("="*60)
print(f"  Records    : {len(dataset):,}")
print(f"  Steps/ep   : {_spe:,}"); print(f"  Epochs     : {CONFIG['num_train_epochs']}")
print(f"  LoRA rank  : {CONFIG['lora_r']} (CPT was 64)")
print(f"  LR         : {CONFIG['learning_rate']} (CPT was 2e-4)")
print(f"  Packing    : False"); print(f"  Resume     : {resume_checkpoint or 'scratch'}")
print("="*60); print("Starting SFT...")

train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
print("="*60); print("SFT COMPLETE"); print("="*60)
print(f"  Steps : {train_result.global_step}")
print(f"  Loss  : {train_result.training_loss:.4f}")
print(f"  Time  : {train_result.metrics.get('train_runtime',0)/60:.1f} min")
'''

CELL8 = '''# Cell 8 — Save Adapter + Report
import os, json as _json

model.save_pretrained(ADAPTER_DIR); tokenizer.save_pretrained(ADAPTER_DIR)
print(f"Adapter saved → {ADAPTER_DIR}")
for f in sorted(os.listdir(ADAPTER_DIR)):
    print(f"  {f:<42} {os.path.getsize(f'{ADAPTER_DIR}/{f}')/1024**2:>8.2f} MB")

rpt = {
    "stage": "04_sft", "base": CONFIG["base_model_id"],
    "cpt":   CONFIG["cpt_adapter_id"],
    "lora_r": CONFIG["lora_r"], "records": len(dataset),
    "loss":  train_result.training_loss,
    "steps": train_result.global_step,
    "min":   train_result.metrics.get("train_runtime",0)/60,
}
with open(f"{LOG_DIR}/stage04_sft_report.json","w") as f: _json.dump(rpt, f, indent=2)
print(f"Report → {LOG_DIR}/stage04_sft_report.json")
tok = os.environ.get("HF_TOKEN BEST", "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")
if CONFIG["push_to_hub"] and tok:
    model.push_to_hub(CONFIG["hf_model_id"], token=tok)
    tokenizer.push_to_hub(CONFIG["hf_model_id"], token=tok)
    print(f"Final adapter pushed → {CONFIG['hf_model_id']}")
'''

CELL9 = '''# Cell 9 — Inference Test: model giờ trả lời CÂU HỎI, không nối văn bản
from unsloth import FastLanguageModel
import torch

FastLanguageModel.for_inference(model)
EOS_ID = tokenizer.convert_tokens_to_ids("<|im_end|>")

def ask(question, context=""):
    user_msg = question + ("\\n\\nNguon: " + context if context else "")
    prompt = (
        "<|im_start|>system\\n" + SYSTEM_PROMPT + "<|im_end|>\\n"
        "<|im_start|>user\\n" + user_msg + "<|im_end|>\\n"
        "<|im_start|>assistant\\n"
    )
    inp = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=200, temperature=0.3,
                              top_p=0.9, repetition_penalty=1.1, do_sample=True,
                              pad_token_id=tokenizer.eos_token_id, eos_token_id=EOS_ID)
    return tokenizer.decode(out[0][inp["input_ids"].shape[1]:],
                              skip_special_tokens=True).split("<|im_end|>")[0].strip()

TESTS = [
    "Dieu kien ket hon theo phap luat Viet Nam la gi?",
    "Thoi han tam giam trong vu an hinh su la bao lau?",
    "Nguoi lao dong co quyen don phuong cham dut hop dong lao dong khong?",
]
print("="*60); print("SFT INFERENCE TEST"); print("="*60)
for i, q in enumerate(TESTS, 1):
    ans = ask(q)
    print(f"\\n[Q{i}] {q}"); print(f"[A]  {ans[:400]}"); print("-"*60)
print("\\nMoi so sanh voi CPT: model gio tra loi dung format Q&A!")
'''

CELL10 = '''# Cell 10 — Evaluation on eval_gold.jsonl (100 samples)
import json as _json, os, torch

if not os.path.exists(EVAL_GOLD):
    print(f"[SKIP] {EVAL_GOLD} not found")
else:
    samples = []
    with open(EVAL_GOLD,"r",encoding="utf-8") as f:
        for line in f:
            try: samples.append(_json.loads(line))
            except: pass
    N = 100
    correct = 0
    print(f"Evaluating {N} samples...")
    for s in samples[:N]:
        pred = ask(s.get("question",""), s.get("context","")[:500]).lower()
        ref  = s.get("answer","").strip().lower()
        if any(w in pred for w in ref.split()[:5] if len(w) > 3):
            correct += 1
    acc = correct / N
    print(f"\\n  Accuracy (soft match): {acc:.1%}  ({correct}/{N})")
    if acc >= 0.75: print("  EXCELLENT — Ready for Stage 5 (DPO)")
    elif acc >= 0.60: print("  GOOD — Ready for Stage 5 (DPO)")
    else: print("  WARN — Consider more SFT data or epochs")

    rp = f"{LOG_DIR}/stage04_sft_report.json"
    if os.path.exists(rp):
        with open(rp) as f: r = _json.load(f)
        r["eval_accuracy"] = acc; r["eval_n"] = N
        with open(rp,"w") as f: _json.dump(r, f, indent=2)
    print(f"\\nDone! Adapter: {CONFIG['hf_model_id']}")
    print("Next: Stage 5 — DPO/RLHF preference alignment")
'''

CELLS = [
    md("# Stage 4 — Supervised Fine-Tuning (SFT)\n\n"
       "**Mục tiêu:** Dạy model trả lời câu hỏi pháp luật theo format Q&A chuẩn.\n"
       "Load CPT adapter (Stage 3) → Merge → LoRA mới → SFT với ChatML template.\n\n"
       "**Auto-Resume:** Cell 6 tự đọc `latest_step.txt` từ HF Hub → resume không cần thao tác.\n\n"
       "**Chạy:** Bấm **Save Version → Save & Run All**. Cả 2 lần đều dùng lệnh này."),
    code(CELL0), code(CELL1), code(CELL2), code(CELL3), code(CELL4),
    code(CELL5), code(CELL6), code(CELL7), code(CELL8), code(CELL9), code(CELL10),
]

nb = {
    "metadata": {
        "kernelspec": {"language":"python","display_name":"Python 3","name":"python3"},
        "language_info": {"name":"python","version":"3.12.12"},
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "dataSources": [{"sourceType":"datasetVersion","sourceId":15011410,
                              "datasetId":9608703,"databundleVersionId":15887829}],
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
