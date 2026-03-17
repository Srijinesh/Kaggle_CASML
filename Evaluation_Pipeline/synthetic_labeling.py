import sys
import os
import json
from tqdm import tqdm

# Ensure parent directory is in path for RAG_Pipeline imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from RAG_Pipeline.main_rag_langgraph import app

def generate_synthetic_labels(queries_path: str, output_path: str):
    """
    Generates a synthetic "Ground Truth" for each query using the LLM.
    This is used for RAGAS evaluation when real ground truth is missing.
    """
    if not os.path.exists(queries_path):
        print(f"Error: Queries file not found at {queries_path}")
        return

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    synthetic_set = []
    print(f"Generating synthetic labels for {len(queries)} questions...")

    for item in tqdm(queries):
        question = item["question"]
        query_id = item["query_id"]

        # To generate a 'Golden Answer', we run the pipeline but with higher scrutiny
        # (e.g., higher max_retries or specific prompt forcing accuracy)
        initial_state = {
            "question": question,
            "documents": [],
            "reranked_docs": [],
            "generation": "",
            "feedback": "",
            "passed_evaluation": False,
            "retries": 0,
            "max_retries": 5 # More retries for "Golden" answers
        }

        try:
            # We assume the self-correction loop will produce a high-quality answer
            final_state = app.invoke(initial_state)
            
            synthetic_set.append({
                "query_id": query_id,
                "question": question,
                "ground_truth": final_state["generation"]
            })
        except Exception as e:
            print(f"Error generating label for {query_id}: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(synthetic_set, f, indent=4, ensure_ascii=False)
        
    print(f"Synthetic test set saved to {output_path}")

if __name__ == "__main__":
    QUERIES_FILE = os.path.join(PARENT_DIR, "Data", "queries.json")
    OUTPUT_FILE = os.path.join(PARENT_DIR, "Evaluation_Pipeline", "synthetic_test_set.json")
    generate_synthetic_labels(QUERIES_FILE, OUTPUT_FILE)
