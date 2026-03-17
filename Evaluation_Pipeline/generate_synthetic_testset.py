import os
import json
import torch
import asyncio
from typing import List
from tqdm import tqdm
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings, ChatHuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# Patching HuggingFacePipeline to support async calls via synchronous fallback
# This is required because Ragas 0.4 relies on async methods which HF Pipeline lacks.
async def _agenerate_patch(self, *args, **kwargs):
    return await asyncio.to_thread(self._generate, *args, **kwargs)

HuggingFacePipeline._agenerate = _agenerate_patch

# Set paths
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PARENT_DIR, "Data")
CONFIG_PATH = os.path.join(PARENT_DIR, "RAG_Pipeline", "config.json")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def get_judge_llm(model_id="Qwen/Qwen2.5-3B-Instruct"):
    print(f"Loading Judge LLM: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
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
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1
    )
    
    return HuggingFacePipeline(pipeline=pipe)

def get_embeddings(model_id="sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_id)

def generate_testset(records_path, output_path, test_size=20):
    with open(records_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    
    # Ragas TestsetGenerator expects Langchain Documents
    from langchain_core.documents import Document
    documents = [
        Document(
            page_content=r["context"], 
            metadata={
                "section": r["metadata"]["section"],
                "page": r["metadata"]["page"],
                "start_page": r["metadata"]["page"], # Added for consistency with LocalLLMGenerator
                "end_page": r["metadata"]["page"],   # Added for consistency with LocalLLMGenerator
                "heading": r["metadata"]["heading"]
            }
        ) for r in records[:20] # Subset for verification
    ]
    
    generator_llm_raw = get_judge_llm()
    # ChatHuggingFace provides a better interface for Ragas than raw pipeline
    generator_llm = ChatHuggingFace(llm=generator_llm_raw)
    critic_llm = generator_llm 
    embeddings = get_embeddings()
    
    from ragas import RunConfig
    run_config = RunConfig(max_workers=2) # Balanced for local RTX 4070
    
    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings
    )
    
    print(f"Generating synthetic testset of size {test_size}...")
    dataset = generator.generate_with_langchain_docs(
        documents, 
        testset_size=test_size,
        run_config=run_config
    )
    
    # Save as JSON for our evaluator
    df = dataset.to_pandas()
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
            
    df.to_json(output_path, orient="records", indent=4)
    print(f"Synthetic testset saved to {output_path}")

if __name__ == "__main__":
    RECORDS_FILE = os.path.join(DATA_DIR, "kaggle_section_records.json")
    OUTPUT_FILE = os.path.join(DATA_DIR, "synthetic_test_set_v2.json")
    
    generate_testset(RECORDS_FILE, OUTPUT_FILE, test_size=2) 
