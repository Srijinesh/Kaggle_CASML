# Evaluation Pipeline

This folder contains the scripts for processing batch queries, generating competition submissions, and calculating RAGAS evaluation metrics locally.

## Scripts Overview

### 1. `generate_submission.py`
**Purpose**: Generates the final `submission.csv` file required for the Kaggle competition.
- **Input**: `Data/queries.json`
- **Output**: `submission.csv` (Root Directory)
- **Usage**:
  ```bash
  python Evaluation_Pipeline/generate_submission.py
  ```

### 2. `synthetic_labeling.py`
**Purpose**: Generates a "Synthetic Gold Standard" answer for each query. This is necessary to calculate **Answer Correctness** when real ground-truth labels are unavailable.
- **Input**: `Data/queries.json`
- **Output**: `Evaluation_Pipeline/synthetic_test_set.json`
- **Usage**:
  ```bash
  python Evaluation_Pipeline/synthetic_labeling.py
  ```

### 3. `ragas_evaluator.py`
**Purpose**: calculates RAGAS metrics (Faithfulness, Relevance, Precision, Correctness) using the local LLM and Embeddings.
- **Input**: `Evaluation_Pipeline/synthetic_test_set.json`
- **Output**: `ragas_eval_report.csv` (Root Directory)
- **Requirements**: Requires `ragas` and `datasets` libraries.
- **Usage**:
  ```bash
  python Evaluation_Pipeline/ragas_evaluator.py
  ```

## Recommended Workflow

1. **Step 1**: Run `synthetic_labeling.py` to create your baseline reference answers.
2. **Step 2**: Run `ragas_evaluator.py` to see your metrics. Use these metrics to tune your `config.json` (e.g., change `retriever_top_k` or `max_retries`).
3. **Step 3**: Once satisfied with the scores, run `generate_submission.py` to create your final file.
