import os
import sys
import json

# Ensure parent directory is in path for RAG_Pipeline imports
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from RAG_Pipeline.retriever import ChromaRetriever

def inspect_metadata():
    retriever = ChromaRetriever(config_path="config.json")
    query = "What is psychology?"
    docs = retriever.retrieve(query)
    
    print(f"\nRetrieved {len(docs)} documents.")
    for i, doc in enumerate(docs):
        print(f"\n--- Doc {i+1} ---")
        print(f"Metadata Keys: {list(doc.metadata.keys())}")
        print(f"Metadata Values: {doc.metadata}")
        print(f"Content Prefix: {doc.page_content[:100]}...")

if __name__ == "__main__":
    inspect_metadata()
