import os
import json
from typing import List
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class ChromaRetriever:
    def __init__(self, config_path: str = "config.json"):
        """
        Initializes the retriever using settings from config.json.
        """
        # Resolve config path relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(config_path):
            config_path = os.path.join(script_dir, config_path)
            
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        # Resolve the db_dir relative to the script dir so it always finds Vector_db
        db_dir = self.config["db_dir"]
        if not os.path.isabs(db_dir):
            db_dir = os.path.normpath(os.path.join(script_dir, db_dir))
            
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config["embedding_model"])
        
        if not os.path.exists(db_dir):
            raise FileNotFoundError(f"Chroma DB directory not found at: {db_dir}. Please ensure the initialization notebook has run.")
            
        self.vectorstore = Chroma(persist_directory=db_dir, embedding_function=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.config["retriever_top_k"]})
        
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieves the most semantically similar chunks.
        """
        return self.retriever.get_relevant_documents(query)

if __name__ == "__main__":
    retriever = ChromaRetriever()
    q = "What are the psychological effects of sleep deprivation?"
    docs = retriever.retrieve(q)
    print(f"Retrieved {len(docs)} docs for query: '{q}'")
