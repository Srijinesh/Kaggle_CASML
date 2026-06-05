import torch
from transformers import pipeline, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline
import time

print("Testing isolated Local LLM generation...")
print(f"CUDA Available: {torch.cuda.is_available()}")

model_id = "Qwen/Qwen2.5-3B-Instruct"

print(f"Loading tokenizer...")
start = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(f"Tokenizer loaded in {time.time() - start:.1f}s")

print(f"Loading model pipeline...")
start = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Explicitly use float32 if CPU to avoid mixed precision hangs
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1,
    torch_dtype=dtype,
    max_new_tokens=50
)
print(f"Model loaded in {time.time() - start:.1f}s")

llm = HuggingFacePipeline(pipeline=pipe)
print("Running inference test...")
res = llm.invoke("Hello, are you working fast?")
print("Result:", res)
print("Success!")
