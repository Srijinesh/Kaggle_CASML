from typing import List, Tuple
from langchain_core.documents import Document
import re

class LlmEvaluator:
    def __init__(self, llm_pipeline):
        """
        Takes the loaded LocalLLMGenerator instance so we don't have to load a massive
        3B parameter model into memory twice.
        """
        self.generator = llm_pipeline
        
    def evaluate_response(self, question: str, response: str, context_docs: List[Document]) -> Tuple[bool, str]:
        """
        Acts as an LLM-as-a-Judge. 
        Returns (True/False passed, Feedback Critique)
        """
        
        context_str = "\n".join([f"- {doc.page_content}" for doc in context_docs])
        
        # We need a highly constrained prompt so the small 3B model doesn't hallucinate during its own evaluation
        evaluation_prompt = f"""<|im_start|>system
You are a ruthless, expert grading assistant evaluating an AI response.
Your ONLY job is to verify if the AI Response correctly answers the Question using ONLY the provided Context Information.

If the response invents facts not in the context, or if it missed a crucial part of the user's question, it FAILS.

You MUST respond strictly in the following JSON-like format. No other text is permitted.
STATUS: [PASS or FAIL]
CRITIQUE: [One sentence explaining why it passed or what it needs to fix in a retry]
<|im_end|>
<|im_start|>user
Context Information:
{context_str}

Question: {question}

AI Response to evaluate: 
{response}
<|im_end|>
<|im_start|>assistant
STATUS:"""

        # Execute the model via the shared pipeline
        raw_output = self.generator.llm.invoke(evaluation_prompt)
        
        # Isolate the model's response part to avoid matching prompt instructions
        if "<|im_start|>assistant" in raw_output:
            processed_output = raw_output.split("<|im_start|>assistant")[-1].strip()
        else:
            processed_output = raw_output.strip()
            
        # Parse the structured output
        passed = False
        critique = "Critique could not be parsed."
        
        if "STATUS: PASS" in processed_output.upper():
            passed = True
        elif "STATUS: FAIL" in processed_output.upper():
            passed = False
            
        critique_match = re.search(r'CRITIQUE:(.*)', processed_output, re.IGNORECASE | re.DOTALL)
        if critique_match:
            critique = critique_match.group(1).strip()
            
        return passed, critique
