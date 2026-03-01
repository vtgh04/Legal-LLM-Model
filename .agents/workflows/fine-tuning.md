---
description: Fine-tuning workflow for the Legal LLM using QLoRA/LoRA/DPO
---

# Fine-Tuning Workflow

This workflow covers running supervised fine-tuning (SFT) and preference
optimization (DPO/ORPO) on a base LLM for the legal domain.

## Pre-requisites

- Processed dataset in `data/processed/final/`
- Base model downloaded to `models/pretrained/`
- Config YAML prepared in `configs/`
- W&B or MLflow set up for tracking

## Steps

1. **Prepare the config file** for your training run. Copy the template and edit it.
   ```bash
   cp configs/sft_template.yaml configs/sft_run_001.yaml
   # Edit model path, dataset path, LoRA rank, learning rate, etc.
   ```

2. **Run Stage 1: QLoRA Supervised Fine-Tuning (SFT)**
   ```bash
   python src/training/sft_trainer.py --config configs/sft_run_001.yaml
   ```
   - Checkpoints are saved to `models/checkpoints/`
   - Monitor loss curves in W&B

// turbo
3. **Evaluate the SFT checkpoint** against LegalBench
   ```bash
   python src/evaluation/run_benchmarks.py --model models/checkpoints/sft_run_001/ --benchmark legalbench
   ```

4. **Run Stage 2: DPO / ORPO Preference Optimization** (if SFT score meets threshold ≥ 75%)
   ```bash
   python src/training/dpo_trainer.py --config configs/dpo_run_001.yaml --base-model models/checkpoints/sft_run_001/
   ```

5. **Final evaluation** on all benchmarks (LegalBench, CaseHOLD, CUAD, ContractNLI)
   ```bash
   python src/evaluation/run_benchmarks.py --model models/checkpoints/dpo_run_001/ --benchmark all
   ```

6. **Merge LoRA adapter** into base model weights for deployment
   ```bash
   python scripts/merge_adapter.py --base models/pretrained/ --adapter models/checkpoints/dpo_run_001/ --output models/final/
   ```
