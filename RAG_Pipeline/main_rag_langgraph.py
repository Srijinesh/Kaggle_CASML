import sys
import os
import json
from typing import TypedDict, List
from langchain.schema import Document
from langgraph.graph import StateGraph, END

# Ensures Python can resolve "RAG_Pipeline..." regardless of whether we run 
# `python main_rag_langgraph.py` from inside the folder or from the Kaggle root dir
KAGGLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if KAGGLE_ROOT not in sys.path:
    sys.path.append(KAGGLE_ROOT)

from RAG_Pipeline.retriever import ChromaRetriever
from RAG_Pipeline.reranker import AdaptiveReranker
from RAG_Pipeline.generator import LocalLLMGenerator
from RAG_Pipeline.evaluator import LlmEvaluator

# Load Config
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# 1. Define the Global State that is passed between nodes
class RagState(TypedDict):
    question: str
    documents: List[Document]      # Dense retrieved docs
    reranked_docs: List[Document]  # Cross-encoder filtered docs
    generation: str                # Current LLM Answer
    feedback: str                  # Evaluator's critique
    retries: int                   # Current retry count
    max_retries: int               

# Prepare the globally shared components to save strict VRAM limits
retriever = ChromaRetriever()
reranker = AdaptiveReranker()
llm_model = LocalLLMGenerator()
evaluator = LlmEvaluator(llm_pipeline=llm_model)

# 2. Define Node Functions
def retrieve_node(state: RagState):
    print("\n--- NODE: RETRIEVAL ---")
    docs = retriever.retrieve(state["question"])
    print(f"Retrieved {len(docs)} documents.")
    return {"documents": docs}

def rerank_node(state: RagState):
    print("\n--- NODE: RERANKING ---")
    filtered = reranker.rerank_and_filter(state["question"], state["documents"])
    print(f"Reranker kept {len(filtered)} high-quality chunk(s).")
    return {"reranked_docs": filtered}

def generate_node(state: RagState):
    print("\n--- NODE: GENERATION ---")
    extra_instructions = ""
    if state["retries"] > 0:
        print(f"Regenerating due to critique: {state['feedback']}")
        extra_instructions = f"Previous attempt failed. Critique: {state['feedback']}. Please fix this."
        
    answer = llm_model.generate_answer(state["question"], state["reranked_docs"], extra_instructions)
    return {"generation": answer, "retries": state["retries"] + 1}

def evaluate_node(state: RagState):
    print("\n--- NODE: EVALUATION (LLM-as-a-Judge) ---")
    passed, critique = evaluator.evaluate_response(state["question"], state["generation"], state["reranked_docs"])
    
    print(f"Evaluate Pass = {passed}")
    print(f"Critique: {critique}")
    
    return {"feedback": critique, "passed_evaluation": passed}

# 3. Define Conditional Edges Logic
def decide_to_generate(state: RagState):
    """
    Decides whether the agent loop should stop or retry generation.
    """
    if state.get("passed_evaluation", False):
        print("--- DECISION: ACCEPT ---")
        return "end"
    elif state["retries"] >= state["max_retries"]:
        print("--- DECISION: MAX RETRIES REACHED. FORCING END ---")
        return "end"
    else:
        print("--- DECISION: REJECT. RE-GENERATING ---")
        return "generate"

# 4. Build the Graph
workflow = StateGraph(RagState)

# Add Nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rerank", rerank_node)
workflow.add_node("generate", generate_node)
workflow.add_node("evaluate", evaluate_node)

# Add Edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "generate")
workflow.add_edge("generate", "evaluate")
workflow.add_conditional_edges(
    "evaluate",
    decide_to_generate,
    {
        "generate": "generate",   # Loop back to generator
        "end": END                # Or stop naturally
    }
)

# Compile
app = workflow.compile()

# Example execution if run directly
if __name__ == "__main__":
    q = "What are the psychological effects of sleep deprivation?"
    
    initial_state = {
        "question": q,
        "documents": [],
        "reranked_docs": [],
        "generation": "",
        "feedback": "",
        "retries": 0,
        "max_retries": config["max_retries"]
    }
    
    print(f"Starting Kaggle RAG Execution for Question: '{q}'\n")
    final_state = app.invoke(initial_state)
    
    print("\n================== FINAL ANSWER ==================")
    print(final_state["generation"])
