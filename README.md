# Kaggle_CASML:
This project is a part of a GEN AI Kaggle Hackathon for Retrieval Augmented Generation Based Question Answering. The project was achieved by chunking a 750 page pdf textbook on Psychology into Langchain documents and subsequently, chunks which are stored in the FAISS vector data store. Embeddings are done with Huggingface MiniLLM sentence transformer family embedder. Mistral 7B LLM is used for generating answers from the nearest most accurate context extracted through langchain pipelines.
Test questions are stored in queries.json, and pdf parsing and data cleaning code is in the Data folder.
Embedding, chunking and text metadata are studied and performed in the chunking_rag_pipeline notebook in the Chunking RAG Modelling folder.
Testing with the Mistral 7B llm is done in the mistral_7b_testing notebook in the Chunking RAG Modelling folder.


#Testing:
For testing the model and pipeline, run the mistrak_7b_testing notebook, all cells in sequential fashion, with the my_vector_index folder in the same location as the running notebook.

BLEU score results from the competition were achieved at a value of 0.277 from a value of 0.11.


Kaggle Submission Link: https://www.kaggle.com/competitions/casml-generative-ai-hackathon/submissions#

# CASML 2024 – Generative AI Hackathon

![Python](https://img.shields.io/badge/pythonngChain/Kagglecontains our solution for the **CASML 2024 – Generative AI Hackathon**, hosted by AiREX at IISc Bangalore. The task was to build a **Retrieval-Augmented Generation (RAG)** system that answers psychology questions using the OpenStax Psychology 2e textbook and provides precise section/page citations.

- **Dataset:** OpenStax Psychology (2e) textbook (CC BY 4.0)  
- **Evaluation:** Multi-criteria metric with context precision, answer faithfulness, correctness, and reference accuracy.

## 🎯 Problem Statement

Build a RAG system that:
- Retrieves relevant text from an 800+ page textbook
- Generates accurate, concise answers to psychology queries
- Provides verifiable references (sections and page numbers)
- Uses open-source models ≤ 3 billion parameters
- Does **not** call any external APIs during inference

## 🧠 System Architecture

```text
User Query
   ↓
Text Preprocessing → Chunking with metadata
   ↓
Embedding & Indexing (FAISS)
   ↓
Similarity Search → Top-k Contexts
   ↓
LLM Generation with Prompt Template
   ↓
Answer + JSON References
```

### Core Components

1. **Document Processor**  
   -  Extracts text from PDF, preserves page & section metadata  
   -  Splits into overlapping chunks for retrieval efficiency  

2. **Vector Store (FAISS)**  
   -  Embeds chunks using a sentence-transformer model  
   -  Builds a Flat-index with FAISS for rapid similarity search  

3. **Retrieval Module**  
   -  Performs k-nearest neighbor search  
   -  Filters and ranks chunks based on relevance  

4. **Generation Module**  
   -  Integrates an open-source LLM via Hugging Face Pipeline  
   -  Uses a custom prompt to ground answers in retrieved context  
   -  Extracts and formats JSON references from chunk metadata  

## 🔧 Installation & Setup

```bash
git clone https://github.com/Srijinesh/Kaggle_CASML.git
cd Kaggle_CASML

python -m venv rag_env
source rag_env/bin/activate    # Windows: rag_env\Scripts\activate

pip install -r requirements.txt
```

**Key Dependencies**  
- langchain, transformers, torch  
- sentence-transformers, faiss-cpu  
- pypdf, pandas, numpy, scikit-learn  
- tqdm, python-dotenv, pydantic


## 💻 Core Usage

1. **Process Textbook & Build FAISS Index**

```python
from src.data_processing.document_processor import TextbookProcessor
from src.embedding.vector_store_faiss import VectorStoreFAISS

processor = TextbookProcessor(chunk_size=1000, chunk_overlap=200)
docs = processor.extract_text_from_pdf("data/raw/psychology_textbook.pdf")
chunks = processor.process_documents(docs)

vs = VectorStoreFAISS(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = vs.create_vectorstore(chunks)
```

2. **Initialize RAG System & Generate Answers**

```python
from src.generation.rag_system import RAGSystem

rag = RAGSystem(
    model_name="llama-2-3b",
    vector_store=vectorstore
)

response = rag.generate_answer("What are the stages of sleep?")
print("Answer:", response["answer"])
print("References:", response["references"])
```

3. **Generate Submission File**

```python
from src.generation.submission_generator import SubmissionGenerator

sub_gen = SubmissionGenerator(rag, "data/raw/queries.json")
submission_df = sub_gen.generate_submission("data/submissions/submission.csv")
sub_gen.validate_submission("data/submissions/submission.csv")
```

## 📊 Evaluation Criteria

1. **Context Precision (20%)**  
   Relevance of retrieved passages to each query  
2. **Answer Faithfulness (20%)**  
   Alignment between generated answer and context  
3. **Answer Correctness (40%)**  
   Accuracy vs. ground-truth answers  
4. **Reference Accuracy (20%)**  
   Precision of section/page citations  

## 📁 Repository Structure

```text
Kaggle_CASML/
├── src/
│   ├── data_processing/
│   ├── embedding/
│   ├── generation/
│   ├── evaluation/
│   └── utils/
├── notebooks/               # Exploratory and demo notebooks
├── data/
│   ├── raw/                 # psychology_textbook.pdf, queries.json
│   ├── processed/           # chunks, embeddings
│   └── submissions/         # submission.csv, validation logs
├── requirements.txt
├── config.yaml
└── README.md
```

## 🚀 Innovations & Challenges

- **Hierarchical Chunking** for optimal context coverage  
- **FAISS Integration** enabling sub-second retrieval  
- **Cross-encoder Re-ranking** (optional) for top-k refinement  
- **Metadata-driven References** ensuring accurate citations  
- **Model Constraint Compliance** with ≤ 3 billion parameters  

## 🛠️ Future Enhancements

- Multi-modal retrieval (including images & tables)  
- Knowledge-graph integration for advanced reasoning  
- Domain-specific fine-tuning for deeper accuracy  
- Real-time API deployment for interactive querying  

## 🤝 Acknowledgments

- **Competition Host:** AiREX, IISc Bangalore  
- **OpenStax:** Psychology 2e textbook (CC BY 4.0)  
- **LangChain, FAISS, Hugging Face Transformers** for essential tooling  
- **Team Members:** Srijinesh (lead), [Your Name] (RAG architect)  

---

*“Bridging psychological knowledge with AI-driven retrieval and generation to empower accurate, cited answers.”*
