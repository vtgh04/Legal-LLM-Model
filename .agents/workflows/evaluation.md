---
description: Evaluation and benchmarking workflow for the Legal LLM
---

# Evaluation & Benchmarking Workflow

This workflow runs all legal domain benchmarks and red-team tests to validate
a model before production promotion.

## Minimum Pass Criteria (all must be met)

| Benchmark | Metric | Minimum |
|---|---|---|
| LegalBench Overall | Accuracy | ≥ 82% |
| CaseHOLD | Macro F1 | ≥ 80% |
| LexGLUE | Macro F1 | ≥ 75% |
| CUAD Contract Review | F1 | ≥ 92% |
| ContractNLI | Accuracy | ≥ 85% |
| Citation Accuracy | Precision | ≥ 95% |
| Hallucination Rate | Rate | ≤ 2% |

## Steps

1. **Set up the evaluation environment**
   ```bash
   pip install -r requirements.txt
   # Ensure model path is set correctly in configs/eval_config.yaml
   ```

// turbo
2. **Run LegalBench** (primary benchmark)
   ```bash
   python src/evaluation/run_benchmarks.py --benchmark legalbench --model models/final/
   ```

// turbo
3. **Run CaseHOLD and LexGLUE**
   ```bash
   python src/evaluation/run_benchmarks.py --benchmark casehold lexglue --model models/final/
   ```

// turbo
4. **Run Contract Benchmarks** (CUAD + ContractNLI)
   ```bash
   python src/evaluation/run_benchmarks.py --benchmark cuad contractnli --model models/final/
   ```

// turbo
5. **Run hallucination detection**
   ```bash
   python src/evaluation/hallucination_eval.py --model models/final/
   ```

6. **Run automated red-teaming** for legal risk scenarios
   ```bash
   python src/evaluation/red_team.py --model models/final/ --scenarios configs/red_team_scenarios.yaml
   ```

7. **Generate evaluation report** (saved to `docs/eval_report.md`)
   ```bash
   python src/evaluation/generate_report.py --output docs/eval_report.md
   ```

8. **Decision gate**: If ALL minima are met → promote to production.
   If ANY fail → return to fine-tuning workflow step 4.
