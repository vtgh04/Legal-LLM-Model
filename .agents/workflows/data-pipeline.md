---
description: Full data ingestion, cleaning, and preparation pipeline for legal corpora
---

# Legal Data Pipeline Workflow

This workflow covers ingesting raw legal data, cleaning it, and producing
training-ready datasets for fine-tuning.

## Steps

1. **Download raw data** into `data/raw/`
   ```bash
   python scripts/download_data.py --source courtlistener --output data/raw/
   ```

// turbo
2. **Run de-identification** to strip PII from raw documents
   ```bash
   python src/data_processing/deidentify.py --input data/raw/ --output data/processed/deidentified/
   ```

// turbo
3. **Normalize legal citations** (Bluebook/OSCOLA)
   ```bash
   python src/data_processing/normalize_citations.py --input data/processed/deidentified/ --output data/processed/normalized/
   ```

// turbo
4. **Deduplicate** the corpus
   ```bash
   python src/data_processing/deduplicate.py --input data/processed/normalized/ --output data/processed/deduped/
   ```

// turbo
5. **Quality filter** by perplexity and length thresholds
   ```bash
   python src/data_processing/quality_filter.py --input data/processed/deduped/ --output data/processed/final/
   ```

6. **Verify** dataset stats — check token count, class distribution, and sample 5 random examples.
   ```bash
   python src/data_processing/verify_dataset.py --input data/processed/final/
   ```
