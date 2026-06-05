import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from typing import List
from langchain_core.documents import Document

class LocalLLMGenerator:
    def __init__(self, config_path: str = "config.json"):
        """
        Initializes a strictly local <3B parameter LLM via HuggingFace for generation.
        Reads model pointers and hyperparameters from the configuration file.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(config_path):
            config_path = os.path.join(script_dir, config_path)
            
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        model_id = self.config["generator_model"]
        
        device_pref = self.config.get("device", "cpu").lower()
        use_cuda = (device_pref == "cuda" and torch.cuda.is_available())
        device_str = "CUDA VRAM (4-bit)" if use_cuda else "CPU RAM"
        print(f"Loading Local LLM: {model_id} into {device_str}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        if use_cuda:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            # Load directly into CPU space. Quantization disabled.
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="cpu",
                torch_dtype=torch.float32
            )
        
        # Utilize hardware acceleration when possible
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.config["max_new_tokens"],
            do_sample=True,
            temperature=self.config["temperature"], 
            top_p=0.9
        )
        
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        
    def generate_answer(self, question: str, context_docs: List[Document], extra_instructions: str = "") -> str:
        """
        Formats a strict prompt demanding grounded citations.
        """
        if not context_docs:
            return "I could not find any relevant information to answer this question."
            
        context_str = ""
        for i, doc in enumerate(context_docs, 1):
            start = doc.metadata.get('start_page', 'Unknown')
            end = doc.metadata.get('end_page', 'Unknown')
            page_str = f"Page {start}" if start == end else f"Pages {start}-{end}"
            
            context_str += f"[Source {i} | {page_str}]:\n{doc.page_content}\n\n"
            
        prompt = f"""<|im_start|>system
You are a highly precise, authoritative psychological assistant. You must answer the user's question using ONLY the provided context blocks below. 

CRITICAL INSTRUCTIONS:
1. You must explicitly cite the page numbers in your answer using the exact format provided in the sources (e.g., "[Page 41]" or "[Pages 101-102]").
2. If the context does not contain the answer, explicitly state "The provided text does not contain information to answer this question."
3. Do not invent or hallucinate information outside the context.
{extra_instructions}
<|im_end|>
<|im_start|>user
Context Information:
{context_str}

Question: {question}
<|im_end|>
<|im_start|>assistant
"""
        result = self.llm.invoke(prompt)
        
        if "<|im_start|>assistant" in result:
            answer = result.split("<|im_start|>assistant")[-1].strip()
        else:
            answer = result.strip()
            
        return answer
