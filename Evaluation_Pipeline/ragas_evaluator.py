import sys
import os
import json
import re
import torch
import pandas as pd
from typing import List, Optional
from tqdm import tqdm
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# Ensure parent directory is in path for RAG_Pipeline imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from RAG_Pipeline.main_rag_langgraph import app
from RAG_Pipeline.generator import LocalLLMGenerator
from RAG_Pipeline.retriever import ChromaRetriever

try:
    from ragas import evaluate, SingleTurnSample
    from ragas.metrics import (
        faithfulness,
        answer_relevance,
        context_precision,
    )
    from ragas.metrics.collections import ContextPrecision, Faithfulness, AnswerCorrectness
    from ragas.llms import LangchainLLMWrapper
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

class ReferenceAccuracy:
    """
    Custom metric to evaluate Reference Accuracy (Kaggle criteria #4).
    Checks if cited page numbers exist in the retrieved context metadata.
    """
    def score(self, sample: SingleTurnSample) -> float:
        answer = sample.response
        retrieved_contexts = sample.retrieved_contexts
        
        # 1. Extract cited pages from answer (e.g., [Page 41] or [Pages 101-102])
        cited_pages = set()
        # Single page pattern
        single_matches = re.findall(r"\[Page\s+(\d+)\]", answer)
        cited_pages.update(single_matches)
        # Range pattern
        range_matches = re.findall(r"\[Pages\s+(\d+)-(\d+)\]", answer)
        for start, end in range_matches:
            for p in range(int(start), int(end) + 1):
                cited_pages.add(str(p))
        
        if not cited_pages:
            # If the answer states no info found, and it's true, score 1.0 (or handle specifically)
            if "The provided text does not contain information" in answer:
                return 1.0
            return 0.0 # No citations provided but answer generated
            
        # 2. Extract available pages from retrieved context metadata
        # Since Ragas SingleTurnSample might just have strings in retrieved_contexts,
        # we might need to pass metadata separately or ensure strings contain page info.
        # However, for this custom class, we'll assume we pass metadata in the 'metadata' dict if needed.
        # But wait, SingleTurnSample doesn't natively hold doc metadata.
        # We'll use a trick: during evaluation loop, we store the metadata.
        
        # This is a placeholder for the Ragas SingleTurnSample interface
        # In this specific RAG pipeline, we calculate it in the loop via calculate_score
        return 0.0 
        
    def calculate_score(self, answer: str, retrieved_docs: List) -> float:
        cited_pages = set()
        single_matches = re.findall(r"\[Page\s+(\d+)\]", answer, re.IGNORECASE)
        cited_pages.update(single_matches)
        range_matches = re.findall(r"\[Pages\s+(\d+)-(\d+)\]", answer, re.IGNORECASE)
        for start, end in range_matches:
            for p in range(int(start), int(end) + 1):
                cited_pages.add(str(p))
                
        if not cited_pages:
            return 0.0
            
        available_pages = set()
        for doc in retrieved_docs:
            try:
                start = int(doc.metadata.get("start_page") or doc.metadata.get("page") or 0)
                end = int(doc.metadata.get("end_page") or start)
                if start > 0:
                    for p in range(start, end + 1):
                        available_pages.add(str(p))
            except (ValueError, TypeError):
                pg = doc.metadata.get("page")
                if pg:
                    available_pages.add(str(pg))
        
        if not available_pages:
            return 0.0
            
        correct_citations = cited_pages.intersection(available_pages)
        precision = len(correct_citations) / len(cited_pages)
        return precision

class RagasEvaluator:
    def __init__(self, judge_model_id="Qwen/Qwen2.5-7B-Instruct"):
        if not RAGAS_AVAILABLE:
            print("Warning: Ragas or datasets library not found.")
            return
            
        self.llm_generator = LocalLLMGenerator()
        self.retriever = ChromaRetriever()
        self.reference_accuracy_metric = ReferenceAccuracy()
        
        # Load Judge Model (7B)
        print(f"Loading Judge LLM for evaluation: {judge_model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(judge_model_id)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            judge_model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1
        )
        self.judge_llm = HuggingFacePipeline(pipeline=pipe)
        
    def run_evaluation(self, test_data_path: str):
        if not RAGAS_AVAILABLE: return

        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        formatted_data = {
            "question": [],
            "contexts": [],
            "answer": [],
            "ground_truth": [],
            "reference_accuracy": []
        }

        print(f"Running inference on {len(test_data)} samples...")
        for item in tqdm(test_data):
            question = item.get("question") or item.get("user_input")
            ground_truth = item.get("ground_truth") or item.get("reference")
            
            # Run RAG pipeline
            res = app.invoke({"question": question, "retries": 0, "documents": [], "reranked_docs": [], "generation": "", "feedback": "", "passed_evaluation": False, "max_retries": 1})
            
            answer = res["generation"]
            retrieved_docs = res["reranked_docs"]
            
            formatted_data["question"].append(question)
            formatted_data["contexts"].append([doc.page_content for doc in retrieved_docs])
            formatted_data["answer"].append(answer)
            formatted_data["ground_truth"].append(ground_truth)
            
            # Calculate Reference Accuracy customly
            ref_acc = self.reference_accuracy_metric.calculate_score(answer, retrieved_docs)
            formatted_data["reference_accuracy"].append(ref_acc)

        dataset = Dataset.from_dict(formatted_data)
        
        print("Starting RAGAS scoring (7B Judge)...")
        # Set up Ragas metrics with our judge
        faith = Faithfulness(llm=self.judge_llm)
        correctness = AnswerCorrectness(llm=self.judge_llm)
        # Note: Context precision usually needs reference or response depending on version
        # We'll use the ones that fit our dataset
        
        result = evaluate(
            dataset,
            metrics=[
                faith,
                correctness,
                ContextPrecision(llm=self.judge_llm)
            ]
        )

        df = result.to_pandas()
        # Add our custom metric to the final report
        df["reference_accuracy"] = formatted_data["reference_accuracy"]
        
        # Calculate Kaggle Weighted Score
        # 20% Context Precision | 20% Faithfulness | 40% Answer Correctness | 20% Reference Accuracy
        df["kaggle_weighted_score"] = (
            df["context_precision"] * 0.20 +
            df["faithfulness"] * 0.20 +
            df["answer_correctness"] * 0.40 +
            df["reference_accuracy"] * 0.20
        )
        
        report_path = os.path.join(PARENT_DIR, "ragas_eval_report_v2.csv")
        df.to_csv(report_path, index=False)
        print(f"Evaluation complete. Report saved to {report_path}")
        print("\nMean Scores:")
        print(df[["context_precision", "faithfulness", "answer_correctness", "reference_accuracy", "kaggle_weighted_score"]].mean())

if __name__ == "__main__":
    TEST_DATA = os.path.join(PARENT_DIR, "Data", "synthetic_test_set_v2.json")
    if not os.path.exists(TEST_DATA):
         # Fallback to old one if v2 not generated yet
         TEST_DATA = os.path.join(PARENT_DIR, "Evaluation_Pipeline", "synthetic_test_set.json")
         
    evaluator = RagasEvaluator()
    evaluator.run_evaluation(TEST_DATA)
