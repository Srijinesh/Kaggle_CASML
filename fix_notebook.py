import json

with open("Vector_db/langchain_chromadb_semantic_chunking.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

new_source = """langchain_docs = []
page_marker_re = re.compile(r"\\[PAGE (\\d+)\\]")

import warnings
warnings.filterwarnings("ignore") # Ignore token length warnings from SemanticChunker

# Process a subset initially to demonstrate if needed, or all of them
for doc in annotated_documents:
    text = doc["annotated_text"]
    
    # Split the long section into semantic blocks
    chunks = chunker.split_text(text)
    
    current_page = "Unknown"
    for chunk in chunks:
        # Extract all page numbers mentioned in this specific chunk
        pages_found = [int(p) for p in page_marker_re.findall(chunk)]
        
        if pages_found:
            start_page = str(min(pages_found))
            end_page = str(max(pages_found))
            current_page = end_page
        else:
            start_page = current_page
            end_page = current_page
            
        # Clean the markers out of the final chunk text
        clean_chunk = page_marker_re.sub("", chunk).strip()
        clean_chunk = re.sub(r"\\s+", " ", clean_chunk).strip()
        
        # Skip empty chunks
        if not clean_chunk:
            continue
            
        metadata = {
            "heading": doc["heading"],
            "sub_heading": doc["sub_heading"],
            "start_page": start_page,
            "end_page": end_page
        }
        langchain_docs.append(Document(page_content=clean_chunk, metadata=metadata))

print(f"Generated {len(langchain_docs)} semantic chunks with precise page grounding.")
"""

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and cell.get("source"):
        source = "".join(cell["source"])
        if "langchain_docs = []" in source and "page_marker_re" in source:
            cell["source"] = new_source.splitlines(True)
            print("Successfully updated cell.")

with open("Vector_db/langchain_chromadb_semantic_chunking.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
