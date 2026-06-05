import os
import json
from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

class AdaptiveReranker:
    def __init__(self, config_path: str = "config.json"):
        """
        Initializes a cross-encoder model for highly accurate reranking using config settings.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(config_path):
            config_path = os.path.join(script_dir, config_path)
            
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        import torch
        device_pref = self.config.get("device", "cpu").lower()
        device = "cuda" if (device_pref == "cuda" and torch.cuda.is_available()) else "cpu"
        print(f"Loading Cross-Encoder Reranker to {device.upper()}...")
            
        self.model = CrossEncoder(self.config["reranker_model"], device=device)
        
    def rerank_and_filter(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Reranks retrieved documents in relation to the query. Calculates all differences 
        (gaps) between consecutive document scores. Slices the list at the 
        most extreme drop in relevancy.
        """
        if not documents:
            return []
            
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        
        # Zip and sort descending
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        print("\n--- Reranker Scores ---")
        for i, (doc, score) in enumerate(scored_docs):
            print(f"Doc {i+1} Score: {score:.4f}")
            
        # If there is only one document, return it
        if len(scored_docs) < 2:
            return [scored_docs[0][0]]
            
        # Calculate leaps/gaps between consecutive sorted scores
        max_drop: float = -1.0
        max_drop_idx: int = 0
        
        for i in range(len(scored_docs) - 1):
            current_score = scored_docs[i][1]
            next_score = scored_docs[i+1][1]
            drop = current_score - next_score
            
            if drop > max_drop:
                max_drop = float(drop)
                max_drop_idx = int(i)
                
        idx1 = int(max_drop_idx) + 1
        idx2 = idx1 + 1
        print(f"\n=> MAXIMUM Gap detected between Doc {idx1} and Doc {idx2}: (Drop of {max_drop:.4f})")
        print(f"=> Removing all documents from Rank {idx2} onwards due to noise.")
        
        # We slice at the index of the score *before* the huge drop.
        # Python slicing [:n] takes elements up to index n-1. We want to include max_drop_idx, so slice at max_drop_idx+1
        split_index = idx1
        return [doc for doc, score in scored_docs[:split_index]]

if __name__ == "__main__":
    reranker = AdaptiveReranker()
    dummy_docs = [
        Document(page_content="Sleep deprivation severely impacts cognitive function and mood."),
        Document(page_content="This is somewhat related but not much."),
        Document(page_content="Apples are a great source of fiber and vitamin C.")
    ]
    query = "What happens if I don't sleep?"
    best_docs = reranker.rerank_and_filter(query, dummy_docs)
    print(f"\nFinal Kept Documents: {len(best_docs)}")
