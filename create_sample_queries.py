import os
import json

def create_sample():
    queries_file = "Data/queries.json"
    with open(queries_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Take first 3 queries
    sample = data[:3]
    
    with open("Data/queries_sample.json", "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=4)

if __name__ == "__main__":
    create_sample()
