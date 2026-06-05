# Kaggle CASML Generative AI Hackathon - Offline RAG System

## 1. Project Overview
This repository contains a fully offline, self-correcting **Agentic Retrieval-Augmented Generation (RAG)** system designed explicitly for the Kaggle CASML competition constraints.

- **Constraints:** Models must be $\le$ 3 Billion Parameters, Open Source, and execute completely locally without any external internet API calls.
- **Architecture Flow:** The codebase is cleanly divided into three distinct pipelines—**Data**, **Model**, and **Evaluation**—orchestrated together using a LangGraph state machine.

---

## 2. The Data Pipeline
The Data Pipeline is responsible for preparing the raw textbook content, identifying semantic boundaries, and persisting the knowledge base for offline semantic search.

* **Semantic Chunking:** We break down the massive textbook PDFs into coherent, contextually complete blocks of text, avoiding brutal mid-sentence splits.
* **Page-Level Metadata Grounding:** The parsing logic uses regular expressions to intelligently extract and attach exact `[PAGE X]` markers to every chunk. This ensures the generator can precisely cite its sources later.
* **Dense Vector Embeddings:** We use `sentence-transformers/all-MiniLM-L6-v2` to mathematically represent the semantic meaning of each chunk.
* **Vector Database (ChromaDB):** The embeddings are persisted locally within `Vector_db/chroma_langchain_db`, functioning as our high-speed retrieval backend.

---

## 3. The Model Pipeline
The Model Pipeline handles the extraction of relevant context and the actual synthesis of the answer using heavily optimized local LLMs.

* **Dual-Stage Retrieval (Hybrid Routing):**
  1. **Dense Retrieval:** We perform an initial fast cosine-similarity search against ChromaDB to pull a wide net of the top 10 chunks.
  2. **Cross-Encoder Adaptive Reranking:** Because dense retrieval can be noisy, we pass the chunks into a highly precise Cross-Encoder (`ms-marco-MiniLM-L-6-v2`). 
* **Dynamic Gap Thresholding:** Instead of taking a static Top-K parameter, the reranker loops over the list of scored chunks and calculates the score differences. It slices the context at the *largest mathematical drop* in relevance, aggressively pruning hallucination-inducing noise.
* **Local Generative AI:** We deploy highly efficient, $\le$ 3B parameter models (e.g., `TinyLlama` or `Qwen2.5-3B`). 
* **4-bit Quantization:** To prevent VRAM deadlocks, we utilize `bitsandbytes` to aggressively compress the generative model footprint via `nf4` configurations, dropping memory limits from 7GB+ down to < 2GB. *(The system natively falls back to standard multi-threading CPU execution paths via `config.json` if NVIDIA CUDA drivers are unavailable).*

---

## 4. The Evaluation & Submission Pipeline
A foundational component of this architecture is the **Self-Reflective Evaluation Pipeline**. We don't just generate text; we explicitly verify its accuracy and adherence to competition constraints.

* **LLM-as-a-Judge:** We repurpose our generator LLM (or a larger 7B model) to act as a strict grading authority. It analyzes responses against retrieved context for:
  1. **Faithfulness:** No hallucinations outside the provided context.
  2. **Answer Correctness:** Accuracy compared to "Golden Answers".
  3. **Context Precision:** Relevance of the retrieved chunks.
  4. **Reference Accuracy:** Precision of page citations.
* **Agentic RAG / Self-Correction Loop:** Managed via **LangGraph**, the pipeline handles cyclic routing. If the Evaluator detects a failure, it routes the prompt *back* to the Generator with a critique to fix the mistake.
* **Kaggle Submission Generation:** The `Evaluation_Pipeline/generate_submission.py` script runs full batch inference on the 50 competition queries, producing a `submission.csv` with expanded and sorted page ranges (e.g., `[2, 3, 4, 5]`).

---

## 5. Directory Structure
```text
Kaggle_CASML/
├── RAG_Pipeline/
│   ├── config.json              # Hyperparameters (Models, Top-K, Device Toggle)
│   ├── retriever.py             # ChromaDB semantic dense retrieval
│   ├── reranker.py              # Cross-Encoder with dynamic gap-thresholding
│   ├── generator.py             # Quantized Local Generative AI Pipeline
│   ├── evaluator.py             # Agentic LLM-as-a-Judge for real-time retries
│   └── main_rag_langgraph.py    # Master LangGraph orchestration script
│
├── Evaluation_Pipeline/         # Batch scoring and submission scripts
│   ├── ragas_evaluator.py       # Metrics: Faithfulness, Correctness, Precision
│   ├── generate_submission.py   # Official Kaggle submission generator
│   └── synthetic_labeling.py    # Ground-truth generation for offline scoring
│
├── Vector_db/
│   └── chroma_langchain_db/     # Persisted dataset embeddings
│
└── Data/                        # Competition queries and textbook records
```

## 6. Usage
Toggle between `"cpu"` and `"cuda"` inside `RAG_Pipeline/config.json`. 

**Run a single query:**
```bash
python RAG_Pipeline/main_rag_langgraph.py
```

**Generate Kaggle Submission:**
```bash
python Evaluation_Pipeline/generate_submission.py
```

**Run Offline RAGAS Evaluation:**
```bash
python Evaluation_Pipeline/ragas_evaluator.py
```
