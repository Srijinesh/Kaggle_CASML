import sys
import os
import json
import pandas as pd
from tqdm import tqdm

# Ensure parent directory is in path for RAG_Pipeline imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from RAG_Pipeline.main_rag_langgraph import app

def generate_submission(queries_path: str, output_path: str):
    """
    Runs the full RAG pipeline on all queries and saves results in Kaggle format.
    """
    if not os.path.exists(queries_path):
        print(f"Error: Queries file not found at {queries_path}")
        return

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    results = []
    print(f"Starting submission generation for {len(queries)} queries...")

    for item in tqdm(queries):
        query_id = item["query_id"]
        question = item["question"]

        # Initialize state for LangGraph
        initial_state = {
            "question": question,
            "documents": [],
            "reranked_docs": [],
            "generation": "",
            "feedback": "",
            "passed_evaluation": False,
            "retries": 0,
            "max_retries": 3 # Minimum retries to speed up batch processing
        }

        try:
            # Run the compiled LangGraph app
            final_state = app.invoke(initial_state)

            # Normalize answer and context to be single-line for better CSV compatibility
            reranked_docs = final_state["reranked_docs"]
            answer = " ".join(final_state["generation"].replace("\n", " ").split()).strip()
            context = " ".join([doc.page_content for doc in reranked_docs]).replace("\n", " ").strip()
            context = " ".join(context.split())

            # Build the specific 'references' JSON structure
            # Check for multiple possible metadata keys to be robust
            sections = []
            pages = []
            
            for doc in reranked_docs:
                # Section/Heading
                sect = doc.metadata.get("heading") or doc.metadata.get("section") or "unknown"
                sections.append(sect)
                
                # Page
                pg = doc.metadata.get("start_page") or doc.metadata.get("page") or "unknown"
                pages.append(str(pg))
            
            sections = list(set(sections))
            pages = list(set(pages))
            
            references = {
                "sections": sections,
                "pages": pages
            }

            results.append({
                "ID": query_id,
                "context": context,
                "answer": answer,
                "references": json.dumps(references)
            })

        except Exception as e:
            print(f"Error processing query {query_id}: {e}")
            results.append({
                "ID": query_id,
                "context": "Error during retrieval",
                "answer": "Error during generation",
                "references": json.dumps({"sections": [], "pages": []})
            })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    QUERIES_FILE = os.path.join(PARENT_DIR, "Data", "queries.json")
    OUTPUT_FILE = os.path.join(PARENT_DIR, "submission.csv")
    generate_submission(QUERIES_FILE, OUTPUT_FILE)
