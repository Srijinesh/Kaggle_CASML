import os, sys
# Ensures Python can resolve "RAG_Pipeline..." regardless of whether we run 
# `python main_rag_langgraph.py` from inside the folder or from the Kaggle root dir
KAGGLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if KAGGLE_ROOT not in sys.path:
    sys.path.append(KAGGLE_ROOT)
from RAG_Pipeline.evaluator import LlmEvaluator
from RAG_Pipeline.generator import LocalLLMGenerator
from langchain_core.documents import Document

AI_Response = """
The psychological effects of sleep deprivation include increased levels of anxiety, difficulties in maintaining attention, making decisions, and recalling long-term memories (Brown, 2012; Alhola & Polo-Kantola, 2007). Sleep deprivation can also lead to impairments in cognitive functions and mood changes, such as irritability and depression (Rattenborg, Lesku, Martinez-Gonzalez, & Lima, 2007). These effects can significantly impact an individual's daily functioning and overall quality of life.
"""

Context = [Document(page_content="""
Sleep deprivation can have a wide range of psychological effects, including increased levels of anxiety, difficulties in maintaining attention, making decisions, and recalling long-term memories (Brown, 2012; Alhola & Polo-Kantola, 2007). Sleep deprivation can also lead to impairments in cognitive functions and mood changes, such as irritability and depression (Rattenborg, Lesku, Martinez-Gonzalez, & Lima, 2007). These effects can significantly impact an individual's daily functioning and overall quality of life.
""")]

Question = "What are the psychological effects of sleep deprivation?"

llm_model = LocalLLMGenerator()
evaluator = LlmEvaluator(llm_model)
passed, critique = evaluator.evaluate_response(Question, AI_Response, Context)

print(f"Passed: {passed}")
print(f"Critique: {critique}")
